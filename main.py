import os
import random
import numpy as np
import yaml
from utils.logger import get_logger
from qlearning import train, evaluate
from envs.frozenlake import make_env

# -----------------------------
# 1. Load config from YAML
# -----------------------------
with open("configs/frozenlake.yaml", "r") as f:
    cfg = yaml.safe_load(f)

logger = get_logger("FrozenLake")

# -----------------------------
# 2. Create environment
# -----------------------------
env = make_env(cfg["env_id"], **cfg["env_kwargs"])
logger.info(f"Environment: {cfg['env_id']} with kwargs={cfg['env_kwargs']}")

# Optional seeding
if cfg["seed"] is not None:
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    env.reset(seed=cfg["seed"])
    try:
        env.action_space.seed(cfg["seed"])
        env.observation_space.seed(cfg["seed"])
    except Exception:
        pass

state_space = env.observation_space.n
action_space = env.action_space.n
logger.info(f"States: {state_space}, Actions: {action_space}")

# -----------------------------
# 3. Initialize / load Q-table
# -----------------------------
def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))

os.makedirs("artifacts", exist_ok=True)

try:
    Qtable = np.load(cfg["qtable_path"])
    if Qtable.shape != (state_space, action_space):
        logger.warning(f"Saved Q-table shape {Qtable.shape}, "
                       f"expected {(state_space, action_space)}. Reinit.")
        Qtable = initialize_q_table(state_space, action_space)
    else:
        logger.info(f"Loaded Q-table from {cfg['qtable_path']}")
except FileNotFoundError:
    logger.info("No saved Q-table found. Initializing new table.")
    Qtable = initialize_q_table(state_space, action_space)

# -----------------------------
# 4. Train
# -----------------------------
Qtable = train(
    cfg["n_train_episodes"], cfg["eps_min"], cfg["eps_max"], cfg["decay_rate"],
    env, cfg["max_steps"], Qtable
)
np.save(cfg["qtable_path"], Qtable)
logger.info(f"Saved Q-table to {cfg['qtable_path']}")

# -----------------------------
# 5. Evaluate
# -----------------------------
avg = evaluate(Qtable, env, cfg["n_eval_episodes"], cfg["max_steps"])
logger.info(f"Average reward: {avg:.2f}")

# Save metrics
with open(cfg["metrics_path"], "a") as f:
    f.write(f"{cfg['n_train_episodes']},{avg:.2f}\n")
logger.info(f"Saved metrics to {cfg['metrics_path']}")
