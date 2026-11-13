from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gymnasium import spaces
from omegaconf import OmegaConf
from loco_env import ActionChunkWrapper, ObservationWrapperGym
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import compose_action
from stable_baselines3.dsrl.hierarchy_schedule import make_hierarchy_schedule

ROOT = Path(__file__).resolve().parents[1]


class PrimitiveEnv:
    action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    observation_space = spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)

    def __init__(self, terminal_at=None, timeout=False):
        self.terminal_at, self.timeout = terminal_at, timeout
        self.calls = 0

    def reset(self, **kwargs):
        self.calls = 0
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        self.calls += 1
        done = self.calls == self.terminal_at
        return np.array([self.calls], dtype=np.float32), 1.0, done, {"TimeLimit.truncated": done and self.timeout}


@pytest.mark.parametrize("timeout", [False, True])
def test_chunk_stops_before_post_terminal_steps(timeout):
    primitive = PrimitiveEnv(terminal_at=2, timeout=timeout)
    env = ActionChunkWrapper(primitive, SimpleNamespace(act_steps=4, obs_dim=1))
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    assert primitive.calls == 2
    assert reward == 2.0
    assert bool(terminated) is (not timeout)
    assert bool(truncated) is timeout
    assert info["actual_primitive_steps"] == 2
    assert info["nominal_primitive_steps"] == 4
    assert info["early_termination_within_chunk"]
    np.testing.assert_array_equal(obs, info["terminal_observation"])


def test_wrapper_time_limit_remains_a_truncation():
    env = ActionChunkWrapper(PrimitiveEnv(), SimpleNamespace(act_steps=4, obs_dim=1), max_episode_steps=3)
    env.reset()
    _, reward, terminated, truncated, info = env.step(np.zeros(4))
    assert reward == 3 and not terminated and truncated
    assert info["termination_reason"] == "time_limit"


def test_pretrained_normalization_coordinates(tmp_path):
    path = tmp_path / "normalization.npz"
    np.savez(path, obs_min=np.array([-2.]), obs_max=np.array([2.]),
             action_min=np.array([-3.]), action_max=np.array([5.]))
    env = ObservationWrapperGym(PrimitiveEnv(), path)
    np.testing.assert_allclose(env.normalize_obs(np.array([0.])), np.array([-2.5e-7]), atol=1e-12)
    np.testing.assert_array_equal(env.unnormalize_action(np.array([-1.])), np.array([-3.]))
    np.testing.assert_array_equal(env.unnormalize_action(np.array([1.])), np.array([5.]))


def test_exact_base_and_bound_preserving_residual():
    base = torch.tensor([[-1., 0., 1.]])
    residual = torch.tensor([[9., -9., 9.]], requires_grad=True)
    low, high = torch.full((3,), -1.), torch.full((3,), 1.)
    zero = compose_action(base, residual, 0., low, high)
    assert torch.equal(zero.action_exec, base)
    edited = compose_action(base, residual, 1., low, high)
    assert (edited.action_exec >= low).all() and (edited.action_exec <= high).all()
    edited.action_exec.sum().backward()
    assert torch.isfinite(residual.grad).all()


@pytest.mark.parametrize("task", ["hopper", "walker", "halfcheetah"])
def test_explicit_release_budgets_and_teacher_settings(task):
    cfg = OmegaConf.load(ROOT / "cfg/gym" / (task + ".yaml"))
    assert cfg.total_timesteps == 800000 and cfg.act_steps == 4
    assert cfg.train.rfs_hier_phase_b_steps == 300000
    assert cfg.train.rfs_hier_phase_r_steps == 500000
    assert cfg.train.rfs_hier_qw_teacher_source == "gaussian"
    assert cfg.train.rfs_hier_qw_candidates_per_state == 4
    assert cfg.train.rfs_hier_noise_actor_gradient_clipping is False
    schedule = make_hierarchy_schedule(str(cfg.train.rfs_hier_schedule_profile), n_envs=int(cfg.env.n_envs), overrides={
        "phase_b_steps": 300000, "phase_r_steps": 500000,
        "beta_hold_steps": 50000, "beta_ramp_steps": 50000})
    schedule.validate_budget(int(cfg.total_timesteps))


@pytest.mark.parametrize("task", ["hopper", "walker", "halfcheetah"])
def test_configuration_resolves_outside_a_hydra_run(task):
    import p6_train  # Registers only the numeric config resolvers.
    cfg = OmegaConf.load(ROOT / "cfg/gym" / (task + ".yaml"))
    OmegaConf.resolve(cfg)
    assert cfg.model.network_path == cfg.base_policy_path
    assert cfg.model.device == cfg.device
