import os
import threading
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from simulator import MujocoSimulator

app = FastAPI()

# Construct the absolute path for the model file relative to this script's location.
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "stanford_tidybot", "scene.xml")
model_path = os.path.abspath(model_path)

simulator = MujocoSimulator(model_path=model_path)


def run_simulator():
    """Function to run the Mujoco simulator."""
    simulator.run()


@app.get("/")
def read_root():
    return {"name": "MujocoSimulator"}


@app.get("/cube/set")
def read_cube_list():
    simulator.set_cube_position()
    return {}


@app.get("/cube/get")
def get_cube_position():
    cube_list = simulator.get_cube_position()
    return {"cube_list": cube_list}


class Cmd(BaseModel):
    target: str


@app.post("/cmd/move_to")
def move_to(cmd: Cmd):
    simulator.move_to(cmd.target)
    return {}


@app.post("/cmd/pick_up")
def pick_up(cmd: Cmd):
    simulator.pick_up(cmd.target)
    return {}

@app.get("/cmd/place")
def place():
    simulator.place()
    return {}


if __name__ == "__main__":
    thread = threading.Thread(target=run_simulator)
    thread.daemon = True  # Ensure the thread exits when the main application does
    thread.start()

    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8800, 
        reload=False,
        log_level="info"
    )
