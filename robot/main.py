"""
FastAPI server for the MuJoCo TidyBot simulation.
Exposes REST API endpoints for robot control and cube manipulation.
"""

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
    """Function to run the MuJoCo simulator in a separate thread."""
    simulator.run()


# --- API Endpoints ---

@app.get("/")
def read_root():
    """Get basic server information."""
    return {"name": "MujocoSimulator"}


@app.get("/cube/set")
def set_cube_positions():
    """Randomize cube positions and orientations in the scene."""
    simulator.set_cube_positions()
    return {}


@app.get("/cube/get")
def get_cube_positions():
    """Get current positions and orientations of all cubes."""
    cube_list = simulator.get_cube_positions()
    return {"cube_list": cube_list}


# --- Data Models ---

class Cmd(BaseModel):
    target: str


# --- Robot Control Endpoints ---

@app.post("/cmd/move_to")
def move_to(cmd: Cmd):
    """Move robot base toward a target object, stopping 50cm away."""
    simulator.move_to(cmd.target)
    simulator.wait_for_completion()
    return {"status": "completed", "target": cmd.target}


@app.post("/cmd/pick_up")
def pick_up(cmd: Cmd):
    """Execute a 4-phase pickup sequence for a target object."""
    simulator.pick_up(cmd.target)
    simulator.wait_for_completion()
    return {"status": "completed", "target": cmd.target}

@app.get("/cmd/place")
def place():
    """Release gripper to place the held object."""
    simulator.place()
    simulator.wait_for_completion()
    return {"status": "completed"}

@app.get("/robot/status")
def get_robot_status():
    """Get current robot status and queue information."""
    return {
        "is_busy": simulator.is_busy(),
        "main_queue_length": len(simulator.movement_queue),
        "mobile_queue_length": len(simulator.mobile_robot.movement_queue),
        "current_trajectory": simulator.current_trajectory.type if simulator.current_trajectory else None,
        "mobile_trajectory": simulator.mobile_robot.current_trajectory.type if simulator.mobile_robot.current_trajectory else None
    }


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
