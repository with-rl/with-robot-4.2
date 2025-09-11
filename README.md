# AI-Powered Robot Task Planner

This project is an **AI-Powered Robot Task Planner** that uses Large Language Models to generate optimal movement sequences for a simulated robot. The system analyzes cube positions and generates structured movement plans to group cubes by color while minimizing travel distance.

## System Architecture

The project consists of two main components:

- **Agent** (`agent/`): LangChain + Google Gemini LLM that analyzes cube positions and generates structured movement plans to group cubes by color while minimizing travel distance
- **Robot Simulator** (`robot/`): MuJoCo-based 3D physics simulation with Stanford TidyBot model, exposing FastAPI endpoints for robot control

## Features

- **AI Task Planning**: Large Language Model (Google Gemini) analyzes cube positions and generates optimal movement sequences
- **Structured Output**: Uses Pydantic models to ensure valid JSON responses from LLM with Action/Plan validation
- **LangGraph Routing**: Intelligent routing system that directs requests to query, update, or planning operations
- **Realistic Physics Simulation**: Utilizes MuJoCo to simulate the robot's dynamics and interactions with its environment
- **Advanced Trajectory Control**: Sophisticated trajectory generation and execution system for smooth robot movements
- **Complex Action Sequences**: Multi-phase actions including complete pickup sequences (approach, grasp, lift, hold)
- **Inverse Kinematics (IK)**: SLSQP-based IK solver translates target positions into joint angles
- **Synchronous API Control**: FastAPI server with synchronous operations that wait for command completion
- **Interactive Development**: Jupyter Notebook interface for AI logic development and testing

## Project Structure

### AI Agent Components
- `agent/src/graph.ipynb`: **Primary development interface** - Main LLM planning logic with interactive development
- `agent/src/graph.py`: LangGraph-based robot routing system with Google Gemini integration
- `agent/.env_sample`: Template for GOOGLE_API_KEY configuration

### Robot Simulation Components  
- `robot/simulator.py`: Core MujocoSimulator class handling 3D physics and robot orchestration
- `robot/manipulator.py`: IKSolver, TrajectoryExecutor, and arm control with inverse kinematics
- `robot/mobile.py`: MobileRobot class for independent base movement with trajectory queue
- `robot/main.py`: FastAPI server exposing robot control endpoints on port 8800
- `robot/client.ipynb`: API interaction examples and testing interface

### 3D Model Assets
- `model/stanford_tidybot/scene.xml`: Complete simulation scene definition with robot and environment
- `model/stanford_tidybot/tidybot.xml`: Robot model with 7-DOF arm configuration
- `model/stanford_tidybot/assets/`: STL meshes and visual components for robot model

## Setup and Installation

### Prerequisites
- Python 3.8+
- Google API key for Gemini LLM access

### Environment Setup
1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd with-robot-4.2
    ```

2. **Create and activate virtual environment:**
    ```bash
    python -m venv robot
    source robot/bin/activate
    # On Windows: robot\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure API access:**
    ```bash
    cp agent/.env_sample .env
    # Edit .env and add: GOOGLE_API_KEY="your_google_api_key"
    ```

## How to Run

### Development Workflow

#### Robot Simulation with API
```bash
cd robot && python main.py
# Runs on http://localhost:8800
```

### Alternative Development
```bash
# Runs LangGraph
cd agent && langgraph dev
```

This will:
1. Launch the FastAPI server on `http://localhost:8800`
2. Open a MuJoCo viewer window displaying the robot simulation
3. Enable AI planning through LangGraph routing system

## API Usage

The API provides endpoints for both robot control and AI planning. All robot control endpoints are **synchronous** - they wait for command completion before responding. You can use any HTTP client or the provided notebooks for interaction.

### API Testing Examples

#### Get Cube Positions
```bash
curl http://localhost:8800/cube/get
```

#### Randomize Cube Positions  
```bash
curl http://localhost:8800/cube/set
```

#### Control Robot Movement
```bash
curl -X POST http://localhost:8800/cmd/move_to \
  -H "Content-Type: application/json" \
  -d '{"target": "cube_1"}'
```

#### Pick Up Objects
```bash
curl -X POST http://localhost:8800/cmd/pick_up \
  -H "Content-Type: application/json" \
  -d '{"target": "cube_blue_02"}'
```

#### Check Robot Status
```bash
curl http://localhost:8800/robot/status
# Response: {"is_busy": false, "main_queue_length": 0, ...}
```

### Core API Endpoints

#### Environment Control
- `GET /cube/get`: Returns current cube positions and orientations
- `GET /cube/set`: Randomizes all cube positions

#### Robot Control (Synchronous)
- `POST /cmd/move_to`: Moves robot base toward target (stops 50cm away)
- `POST /cmd/pick_up`: Moves robot arm to pick up specified object
- `GET /cmd/place`: Commands robot to place held object

#### Robot Status
- `GET /robot/status`: Get current robot status and queue information

For a detailed, hands-on guide on using these endpoints, please see the `robot/client.ipynb` notebook.

## Architecture Overview

### AI Planning Architecture
1. **AI Planning**: `agent/src/graph.ipynb` → LangGraph routing → Pydantic Action/Plan models → structured JSON
2. **API Processing**: FastAPI endpoints → MujocoSimulator methods → parallel trajectory queues  
3. **Parallel Execution**: Mobile base and manipulator systems operate independently with separate queues
4. **Synchronous Response**: All robot control commands block until completion before responding

### Core System Components

#### AI Agent Layer
- **LangGraph Router**: Intelligent routing to query/update/plan operations using Google Gemini LLM
- **Pydantic Models**: Structured data validation for Action/Plan objects ensuring valid LLM output
- **Plan Generation**: AI analyzes cube positions and generates optimal movement sequences to group by color

#### Robot Control Layer  
- **MujocoSimulator**: Main orchestrator managing both mobile and manipulator trajectory queues
- **IKSolver**: SLSQP-based inverse kinematics solver with position and orientation cost functions
- **TrajectoryExecutor**: S-curve interpolation system for smooth, physics-based arm movements
- **MobileRobot**: Independent base movement system with dedicated trajectory queue and execution
- **PickupSequence**: Complex 4-phase pickup coordination (approach, grasp, lift, hold)

### Key Data Models
- **Action**: Single robot command (`{"action": "move", "start": "cube_1", "end": "cube_2"}`)
- **Plan**: Sequential list of Actions for complete task execution  
- **Structured LLM Output**: Uses Pydantic to ensure valid JSON responses from Gemini

### Critical Implementation Notes
- **Quaternion Conversion**: Handle MuJoCo [w,x,y,z] ↔ SciPy [x,y,z,w] format differences carefully
- **Joint Control**: 7-DOF arm controlled via `data.ctrl[3:10]` for proper physics-based joint control  
- **Model Paths**: Must use absolute paths for MuJoCo models (automatically handled in main.py)
- **Threading Model**: Simulator runs as daemon thread while FastAPI serves HTTP on port 8800
- **End Effector**: Access robot end effector position through "pinch_site" site in MuJoCo model

### Validation Approach
- No formal test suite - validation through interactive notebook execution
- Visual validation through 3D simulation and matplotlib plots
- API testing through curl commands or robot/client.ipynb