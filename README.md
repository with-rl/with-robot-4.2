# Stanford TidyBot MuJoCo Simulator

This project provides a simulation of the Stanford TidyBot using MuJoCo, a powerful physics simulator. The simulation is controlled via a REST API built with FastAPI, allowing for remote and programmatic control of the robot's actions.

## Features

- **Realistic Physics Simulation**: Utilizes MuJoCo to simulate the robot's dynamics and interactions with its environment.
- **Advanced Trajectory Control**: A sophisticated trajectory generation and execution system enables smooth, predictable movements for the robot's arm and base.
- **Complex Action Sequences**: Supports multi-phase actions, such as a complete pickup sequence involving approaching, grasping, and lifting an object.
- **Inverse Kinematics (IK)**: An IK solver (`IKSolver`) translates target positions and orientations into the required joint angles for the robot's arm.
- **REST API Control**: A FastAPI server exposes endpoints to control the simulation, manage the environment, and command the robot.
- **Interactive Client**: A Jupyter Notebook (`robot/client.ipynb`) provides a practical example of how to interact with the simulation via the API.
- **Dynamic Environment**: The positions of objects (cubes) in the scene can be randomized for dynamic testing scenarios.

## Project Structure

- `robot/simulator.py`: The core of the project, containing the `MujocoSimulator` class that manages the simulation loop, robot state, and trajectory execution.
- `robot/simulator_util.py`: Provides utility classes, most notably the `IKSolver` for inverse kinematics calculations.
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
python robot/main.py
```

This will:
1.  Launch the FastAPI server on `http://localhost:8800`.
2.  Open a MuJoCo viewer window displaying the robot simulation.

## API Usage

The API provides several endpoints to interact with the simulation. You can use any HTTP client or the provided `robot/client.ipynb` notebook to send requests.

### Endpoints

- `GET /`
  - **Description**: Checks the status of the server.

- `GET /cube/set`
  - **Description**: Randomizes the positions of the cubes in the simulation environment.

- `GET /cube/get`
  - **Description**: Retrieves a list of all cubes and their current positions and orientations.

- `POST /cmd/move_to`
  - **Description**: Moves the robot's base to a position near the specified target object.
  - **Body**: `{"target": "cube_name"}`

- `POST /cmd/pick_up`
  - **Description**: Executes a full pickup sequence for the specified target object.
  - **Body**: `{"target": "cube_name"}`

- `GET /cmd/place`
  - **Description**: Commands the robot to place the object it is currently holding.

For a detailed, hands-on guide on using these endpoints, please see the `robot/client.ipynb` notebook.