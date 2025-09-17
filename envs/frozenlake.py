# envs/frozenlake.py
import gymnasium as gym

def make_env(env_id: str, **env_kwargs):
    """Factory for the Gymnasium envs (kept simple on purpose)."""
    env = gym.make(env_id, **env_kwargs)
    return env
