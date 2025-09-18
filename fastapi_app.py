from fastapi import FastAPI
import numpy as np

# Create the app
app = FastAPI()

# Load pretrained Q-table (make sure the file exists in artifacts/)
Q = np.load("artifacts/qtable_frozenlake.npy")

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
