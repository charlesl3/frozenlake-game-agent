import numpy as np
import os
from fastapi import FastAPI

app = FastAPI()

Q_PATH = "artifacts/qtable_frozenlake.npy"

if os.path.exists(Q_PATH):
    Q = np.load(Q_PATH)
else:
    Q = np.zeros((16, 4))  # trivial FrozenLake table (4x4 states, 4 actions)


@app.get("/act")
def get_action(state: int):
    """
    Given a state index, return the best action (argmax of Q-table row).
    Example: GET /act?state=3
    """
    if state < 0 or state >= Q.shape[0]:
        return {"error": f"State {state} out of range (0 to {Q.shape[0]-1})"}
    
    action = int(np.argmax(Q[state]))
    return {"state": state, "action": action}
