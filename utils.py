from transformer_lens import HookedSAEConfig, HookedSAE
from sae_lens import SparseAutoencoder
from sae_lens.training.train_sae_on_language_model import LanguageModelSAERunnerConfig

# convert sl sae to tl sae: https://gist.github.com/dtch1997/e31a7dfdad822a12c8e2ddc272a33d24

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