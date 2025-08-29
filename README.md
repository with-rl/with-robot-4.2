# AI-Powered Robot Task Planner

This project utilizes a Large Language Model (LLM) to generate an optimal sequence of actions for a simulated robot. The primary task for the robot is to rearrange colored cubes in a 2D space to group them by color, minimizing the total travel distance.

## How it Works

The core logic is contained within the `src/graph.ipynb` Jupyter Notebook, which orchestrates the entire process:

1.  **Initialization**: A Google Gemini Pro model is initialized using the LangChain library.
2.  **State Definition**: The initial positions of several colored (red and blue) cubes are defined in a 2D space. The notebook also includes functionality to randomize these positions for different scenarios.
3.  **Structured Output**: Pydantic models (`Action` and `Plan`) are used to ensure the LLM returns a well-structured plan that the system can parse and execute. An `Action` consists of a `move` command with a `start` and `end` location, and a `Plan` is a list of these actions.
4.  **Prompt Engineering**: A detailed prompt is constructed to guide the LLM. It includes:
    *   The robot's role and objective.
    *   Strict rules for generating the plan (e.g., minimize distance, group by color).
    *   The current state of all cubes, formatted as a JSON object.
    *   The required JSON output format for the plan.
5.  **Plan Generation**: The LLM processes the prompt and generates a `Plan` containing the optimal sequence of `Action`s to achieve the goal.
6.  **Visualization**: `matplotlib` is used to create a scatter plot that visualizes the initial positions of the cubes and the final arrangement after the robot executes the generated plan.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd with-robot-4.2
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv robot
    source robot/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory by copying the `.env_sample` file.
    ```bash
    cp .env_sample .env
    ```
    Open the `.env` file and add your Google API Key:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY"
    ```

## Usage

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  Open the `src/graph.ipynb` notebook.
3.  Run the cells sequentially to see the plan generation and visualization.

## Key Dependencies

*   `langchain` & `langgraph`: For building and running the LLM-powered graph.
*   `langchain-google-genai`: For interacting with the Google Gemini models.
*   `pydantic`: For data validation and structured output.
*   `numpy`: For numerical operations.
*   `matplotlib`: For data visualization.
*   `python-dotenv`: For managing environment variables.

## License

This project is licensed under the terms of the LICENSE file.
