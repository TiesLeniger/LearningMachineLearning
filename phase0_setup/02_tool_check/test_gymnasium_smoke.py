"""Gymnasium CartPole: spaces and seeded determinism."""

import gymnasium as gym
import numpy as np


def rollout(seed: int, n_steps: int = 50) -> np.ndarray:
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    traj = [obs]
    for t in range(n_steps):
        obs, _, terminated, truncated, _ = env.step(t % 2)  # fixed action sequence
        traj.append(obs)
        if terminated or truncated:
            break
    env.close()
    return np.array(traj)


def test_spaces():
    env = gym.make("CartPole-v1")
    assert env.observation_space.shape == (4,)
    assert env.action_space.n == 2
    env.close()


def test_seed_reproducible():
    np.testing.assert_array_equal(rollout(0), rollout(0))


def test_seed_changes_initial_state():
    assert not np.array_equal(rollout(0)[0], rollout(1)[0])
