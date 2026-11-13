"""Frozen DPPO decoder interface for locomotion adaptation."""
from _paths import configure_vendor_paths
configure_vendor_paths()
import hydra
import torch


class FrozenDiffusionPolicy:
    def __init__(self, model):
        self.base_policy = model.eval()
        self.base_policy.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, obs, initial_noise, return_numpy=True):
        samples = self.base_policy(
            cond={"state": obs, "noise_action": initial_noise}, deterministic=True
        ).trajectories.detach()
        return samples.cpu().numpy() if return_numpy else samples


def load_base_policy(cfg):
    return FrozenDiffusionPolicy(hydra.utils.instantiate(cfg.model))
