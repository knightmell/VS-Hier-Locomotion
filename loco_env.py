"""D4RL normalization and chunk execution, without manipulation dependencies."""
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTION_CHUNK_EARLY_BREAK = "early_break_on_done"


class ObservationWrapperGym:
    """Apply the pretrained policy's min/max coordinate transforms."""
    def __init__(self, env, normalization_path):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        with np.load(Path(normalization_path)) as values:
            self.obs_min = values["obs_min"].copy()
            self.obs_max = values["obs_max"].copy()
            self.action_min = values["action_min"].copy()
            self.action_max = values["action_max"].copy()

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        seed = seed if seed is not None else (options or {}).get("seed")
        if seed is not None and hasattr(self.env, "seed"):
            self.seed(seed)
            observation = self.env.reset()
        elif seed is not None:
            observation = self.env.reset(seed=seed, options=options)
        else:
            observation = self.env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        return self.normalize_obs(observation)

    def normalize_obs(self, observation):
        return 2 * ((observation - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

    def unnormalize_action(self, action):
        return (np.asarray(action) + 1) / 2 * (self.action_max - self.action_min) + self.action_min

    def step(self, action):
        observation, reward, done, info = self.env.step(self.unnormalize_action(action))
        return self.normalize_obs(observation), reward, done, info

    def get_normalized_score(self, reward):
        return self.env.get_normalized_score(reward)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        return self.env.close()


class ActionChunkWrapper(gym.Env):
    """Execute at most H actions; stop at the first terminal or truncation."""
    def __init__(self, env, cfg, max_episode_steps=1000,
                 action_chunk_termination_semantics=ACTION_CHUNK_EARLY_BREAK):
        if action_chunk_termination_semantics != ACTION_CHUNK_EARLY_BREAK:
            raise ValueError("The locomotion release uses early-break chunk execution")
        self.env = env
        self.act_steps = int(cfg.act_steps)
        self.max_episode_steps = int(max_episode_steps)
        self.action_chunk_termination_semantics = action_chunk_termination_semantics
        self.action_space = spaces.Box(
            low=np.tile(env.action_space.low, self.act_steps),
            high=np.tile(env.action_space.high, self.act_steps), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(int(cfg.obs_dim),), dtype=np.float32)
        self.count = 0

    def reset(self, *, seed=None, options=None):
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options
        try:
            observation = self.env.reset(**kwargs)
        except TypeError:
            kwargs.pop("options", None)
            observation = self.env.reset(**kwargs)
        if isinstance(observation, tuple):
            observation = observation[0]
        self.count = 0
        return observation, {}

    def step(self, action):
        actions = np.asarray(action).reshape(self.act_steps, -1)
        reward_sum = 0.0
        terminated = truncated = False
        terminal_index = None
        for index, primitive in enumerate(actions):
            observation, reward, done, info = self.env.step(primitive)
            info = dict(info)
            self.count += 1
            reward_sum += reward
            timeout = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done and not timeout)
            truncated = bool(timeout or self.count >= self.max_episode_steps)
            if done or truncated:
                terminal_index = index
                break
        actual = index + 1
        info.update({
            "action_chunk_termination_semantics": ACTION_CHUNK_EARLY_BREAK,
            "nominal_primitive_steps": self.act_steps,
            "actual_primitive_steps": actual,
            "early_termination_within_chunk": bool((terminated or truncated) and actual < self.act_steps),
            "termination_primitive_index": terminal_index,
            "termination_reason": "environment_terminal" if terminated else "time_limit" if truncated else None,
        })
        if terminated or truncated:
            info["terminal_observation"] = observation
        return observation, reward_sum, terminated, truncated, info

    def get_normalized_score(self, reward):
        return self.env.get_normalized_score(reward)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
