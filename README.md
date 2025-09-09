# Stanford TidyBot MuJoCo Simulator

This project provides a simulation of the Stanford TidyBot using MuJoCo, a powerful physics simulator. The simulation is controlled via a REST API built with FastAPI, allowing for remote and programmatic control of the robot's actions.

## Features

- **Realistic Physics Simulation**: Utilizes MuJoCo to simulate the robot's dynamics and interactions with its environment.
- **Advanced Trajectory Control**: A sophisticated trajectory generation and execution system enables smooth, predictable movements for the robot's arm and base, with modular mobile and manipulator control.
- **Complex Action Sequences**: Supports multi-phase actions, such as a complete pickup sequence involving approaching, grasping, and lifting an object.
- **Inverse Kinematics (IK)**: An IK solver (`IKSolver`) translates target positions and orientations into the required joint angles for the robot's arm.
- **Synchronous API Control**: A FastAPI server exposes endpoints to control the simulation, with synchronous operation that waits for command completion before responding.
- **Interactive Client**: A Jupyter Notebook (`robot/client.ipynb`) provides a practical example of how to interact with the simulation via the API.
- **Dynamic Environment**: The positions of objects (cubes) in the scene can be randomized for dynamic testing scenarios.

## Project Structure

- `robot/simulator.py`: The core of the project, containing the `MujocoSimulator` class that manages the simulation loop, robot state, and trajectory execution.
- `robot/manipulator.py`: Provides manipulator functionality including trajectory classes, inverse kinematics solver (`IKSolver`), and trajectory executor for arm and gripper control.
- `robot/mobile.py`: Handles mobile robot base movement and trajectory execution, separated for modular robot control.
- `robot/main.py`: A FastAPI application that creates a REST API to control the simulator. It runs the simulator in a separate thread.
- `robot/client.ipynb`: An example Jupyter Notebook demonstrating how to send commands to the simulator using the REST API.
- `model/stanford_tidybot/`: Contains the MuJoCo XML files that define the robot's structure (`tidybot.xml`) and the simulation scene (`scene.xml`).
- `requirements.txt`: A list of all Python dependencies required for the project.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd with-robot-4.2
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

To start the simulation and the API server, run the following command from the project's root directory:

```bash
cd robot
python main.py
mjpython main.py  # on mac
```

This will:
1.  Launch the FastAPI server on `http://localhost:8800`.
2.  Open a MuJoCo viewer window displaying the robot simulation.

## API Usage

The API provides several endpoints to interact with the simulation. All robot control endpoints are **synchronous** - they wait for the command to complete before returning a response. You can use any HTTP client or the provided `robot/client.ipynb` notebook to send requests.

### Endpoints

#### Server Status
- `GET /`
  - **Description**: Checks the status of the server.
  - **Response**: `{"name": "MujocoSimulator"}`

#### Environment Control
- `GET /cube/set`
  - **Description**: Randomizes the positions of the cubes in the simulation environment.
  - **Response**: `{}`

- `GET /cube/get`
  - **Description**: Retrieves a list of all cubes and their current positions and orientations.
  - **Response**: `{"cube_list": [{"name": "cube_1", "position": [x, y, z], "euler": [rx, ry, rz]}, ...]}`

#### Robot Control (Synchronous)
- `POST /cmd/move_to`
  - **Description**: Moves the robot's base to a position near the specified target object. **Waits for completion.**
  - **Body**: `{"target": "cube_name"}`
  - **Response**: `{"status": "completed", "target": "cube_name"}`

- `POST /cmd/pick_up`
  - **Description**: Executes a full pickup sequence for the specified target object. **Waits for completion.**
  - **Body**: `{"target": "cube_name"}`
  - **Response**: `{"status": "completed", "target": "cube_name"}`

- `GET /cmd/place`
  - **Description**: Commands the robot to place the object it is currently holding. **Waits for completion.**
  - **Response**: `{"status": "completed"}`

#### Robot Status
- `GET /robot/status`
  - **Description**: Get current robot status and queue information.
  - **Response**: 
    ```json
    {
      "is_busy": false,
      "main_queue_length": 0,
      "mobile_queue_length": 0,
      "current_trajectory": null,
      "mobile_trajectory": null
    }
    ```

For a detailed, hands-on guide on using these endpoints, please see the `robot/client.ipynb` notebook.

## Architecture Overview

The project follows a modular architecture with clear separation of concerns:

### Core Components

- **MujocoSimulator** (`simulator.py`): Main orchestrator that manages the simulation loop, coordinates between mobile and manipulator systems, and handles trajectory queue management.

- **MobileRobot** (`mobile.py`): Dedicated to mobile base movement with its own trajectory queue and execution system. Handles base positioning and rotation with S-curve interpolation.

- **IKSolver & TrajectoryExecutor** (`manipulator.py`): Handles inverse kinematics calculations and trajectory execution for arm and gripper movements. Supports complex multi-phase pickup sequences.

### Data Flow

1. **API Request** → FastAPI endpoint receives command
2. **Command Queuing** → Simulator adds trajectory to appropriate queue (mobile/manipulator)
3. **Parallel Execution** → Mobile and manipulator trajectories execute independently
4. **Completion Monitoring** → API waits for all queues to complete before responding
5. **Response** → Client receives completion confirmation

### Key Design Decisions

- **Modular Control**: Separate mobile and manipulator systems allow independent operation
- **Synchronous API**: Commands block until completion for predictable client behavior
- **Queue-based Execution**: Multiple trajectories can be queued and executed in sequence
- **Physics-based Control**: Uses MuJoCo's `data.ctrl` for proper joint control instead of direct position setting