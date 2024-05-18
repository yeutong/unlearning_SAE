# %%
import os
import torch
from transformer_lens import HookedTransformer, HookedSAETransformer
from transformer_lens.utils import test_prompt
from sae_lens import SparseAutoencoder, ActivationsStore
from utils import convert_sl_sae_to_tl_sae
from tqdm import tqdm
import plotly.express as px

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


def get_tokens(
    activation_store: ActivationsStore,
    n_batches_to_sample_from: int = 2**10,
    n_prompts_to_select: int = 4096 * 6,
):
    all_tokens_list = []
    pbar = tqdm(range(n_batches_to_sample_from))
    for _ in pbar:
        batch_tokens = activation_store.get_batch_tokens()
        batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
            : batch_tokens.shape[0]
        ]
        all_tokens_list.append(batch_tokens)

    all_tokens = torch.cat(all_tokens_list, dim=0)
    all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
    return all_tokens[:n_prompts_to_select]


all_tokens = get_tokens(activation_store)  # should take a few minutes
all_tokens = all_tokens[:300, :30]
# %%
from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData

test_feature_idx_gpt = list(range(10)) + [14057]

feature_vis_config_gpt = SaeVisConfig(
    hook_point=hook_name,
    features=test_feature_idx_gpt,
    batch_size=2048,
    minibatch_size_tokens=128,
    verbose=True,
)

sae_vis_data_gpt = SaeVisData.create(
    encoder=sparse_autoencoder,
    model=model,
    tokens=all_tokens,  # type: ignore
    cfg=feature_vis_config_gpt,
)


# %%
import os
import webbrowser

for feature in test_feature_idx_gpt:
    filename = f"{feature}_feature_vis_demo_gpt.html"
    sae_vis_data_gpt.save_feature_centric_vis(filename, feature)
    # webbrowser.open(filename)
    # install "live server" extension then click "open with live server"
# %%
