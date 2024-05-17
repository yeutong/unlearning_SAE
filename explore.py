# %%
import torch
from transformer_lens import HookedTransformer
from sae_lens import SparseAutoencoder, ActivationsStore

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gemma-2b")

# %%
sparse_autoencoder = SparseAutoencoder.from_pretrained(
  "gemma-2b-res-jb", # to see the list of available releases, go to: https://github.com/jbloomAus/SAELens/blob/main/sae_lens/pretrained_saes.yaml
  "blocks.0.hook_resid_post" # change this to another specific SAE ID in the release if desired. 
)
activation_store = ActivationsStore.from_config(model, sparse_autoencoder.cfg)
# %%
# from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
# # %%   
# # get sae directory
# sae_directory = get_pretrained_saes_directory()

# # get the repo id and path to the SAE
# if release not in sae_directory:
#     raise ValueError(
#         f"Release {release} not found in pretrained SAEs directory."
#     )
# if sae_id not in sae_directory[release].saes_map:
#     raise ValueError(f"ID {sae_id} not found in release {release}.")
# sae_info = sae_directory[release]
# hf_repo_id = sae_info.repo_id
# hf_path = sae_info.saes_map[sae_id]