# %%
import os
import torch
from transformer_lens import HookedTransformer, HookedSAETransformer
from transformer_lens.utils import test_prompt
from sae_lens import SparseAutoencoder, ActivationsStore

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
model = HookedTransformer.from_pretrained("gemma-2b")

hook_name = "blocks.6.hook_resid_post"
sparse_autoencoder = SparseAutoencoder.from_pretrained(
  "gemma-2b-res-jb", # to see the list of available releases, go to: https://github.com/jbloomAus/SAELens/blob/main/sae_lens/pretrained_saes.yaml
  hook_name, # change this to another specific SAE ID in the release if desired. 
  device=device
)
activation_store = ActivationsStore.from_config(model, sparse_autoencoder.cfg)

# %%
# try generate something
prompts = [
    "Who is Harry Potter?",
    "Harry Potter’s two best friends are",
    "When Harry went back to class, he saw that his best friends,"
    "Ron and Hermione went",
    "The list of major characters from Harry Potter include Snape, Couch and",
    "Harry Potter studies"
]
batch_tokens = model.to_tokens(prompts)

for prompt in prompts:
    print("Prompt: ", prompt)
    print(model.generate(prompt, max_new_tokens=30))
    print()
# %%


test_prompt(
    "Harry Potter’s two best friends are",
    " Ron",
    model,
    prepend_space_to_answer=False,
)
# %%
sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads

with torch.no_grad():
    # activation store can give us tokens.
    batch_tokens = activation_store.get_batch_tokens()
    batch_tokens = batch_tokens[:, :30]
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

    # Use the SAE
    sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
        cache[sparse_autoencoder.cfg.hook_point]
    )

    # save some room
    del cache

    # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
    l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
    print("average l0", l0.mean().item())
    # px.histogram(l0.flatten().cpu().numpy()).show()

# %%

# see transformer hookedSAE demo: https://colab.research.google.com/github/ckkissane/TransformerLens/blob/hooked-sae-transformer/demos/HookedSAETransformerDemo.ipynb#scrollTo=6CRWdyWxtkWU
# convert sl sae to tl sae: https://gist.github.com/dtch1997/e31a7dfdad822a12c8e2ddc272a33d24

from transformer_lens import HookedSAEConfig, HookedSAE
from sae_lens import SparseAutoencoder
from sae_lens.training.train_sae_on_language_model import LanguageModelSAERunnerConfig

def sl_sae_cfg_to_hooked_sae_cfg(
    resid_sae_cfg: LanguageModelSAERunnerConfig,
) -> HookedSAEConfig:
    new_cfg = {
        "d_sae": resid_sae_cfg.d_sae,
        "d_in": resid_sae_cfg.d_in,
        "hook_name": resid_sae_cfg.hook_point,
    }
    return HookedSAEConfig.from_dict(new_cfg)


def convert_sl_sae_to_tl_sae(
    sl_sae: SparseAutoencoder,
) -> HookedSAE:
    state_dict = sl_sae.state_dict()
    # NOTE: sae-lens uses a 'scaling factor'
    # For now, just check this is 1 and then remove it
    assert torch.allclose(
        state_dict["scaling_factor"], torch.ones_like(state_dict["scaling_factor"])
    ), f"Scaling factor {state_dict['scaling_factor']} was not close to 1" 
    state_dict.pop("scaling_factor")

    cfg = sl_sae_cfg_to_hooked_sae_cfg(sl_sae.cfg)
    tl_sae = HookedSAE(cfg)
    tl_sae.load_state_dict(state_dict)
    return tl_sae
# %%
model_sae = HookedSAETransformer.from_pretrained("gemma-2b")
hooked_sae = convert_sl_sae_to_tl_sae(sparse_autoencoder)

# %%
with model_sae.saes(saes=[hooked_sae]):
    # for prompt in prompts:
    #     print("Prompt: ", prompt)
    #     print(model.generate(prompt, max_new_tokens=30))
    #     print()



    # tmp_prompt = "Harry Potter’s two best friends are"
    # test_prompt(
    #     tmp_prompt,
    #     " Ron",
    #     model_sae,
    #     prepend_space_to_answer=False,
    # )

    loss = model_sae(tmp_prompt, return_type='loss')
    print(loss)

# %%
model_sae(tmp_prompt, return_type='loss')


# %%
logits, cache = model.run_with_cache(tmp_prompt, prepend_bos=True)
tokens = model.to_tokens(tmp_prompt)
sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
    cache[sparse_autoencoder.cfg.hook_point]
)
def reconstr_hook(activations, hook, sae_out):
    return sae_out


def zero_abl_hook(mlp_out, hook):
    return torch.zeros_like(mlp_out)


hook_point = sparse_autoencoder.cfg.hook_point

print("Orig", model(tokens, return_type="loss").item())
print(
    "reconstr",
    model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (
                hook_point,
                partial(reconstr_hook, sae_out=sae_out),
            )
        ],
        return_type="loss",
    ).item(),
)
print(
    "Zero",
    model.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(hook_point, zero_abl_hook)],
    ).item(),
)


# %%
# try to understand HookedSAETransformer