"""
AI Robot Task Planner using LangGraph.

Features: Route robot operations to query, update, or execute cube commands.
Flow: START → router → [query_cube/update_cube/plan] → END
"""

import logging
from typing import TypedDict, Literal, List
from typing_extensions import Annotated, NotRequired

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROBOT_API_BASE_URL = "http://localhost:8800"
DEFAULT_TIMEOUT = 60


# Data Models
class RobotRoute(BaseModel):
    """Routing decision model with validation for Robot operations."""
    step: Literal["query_cube", "update_cube", "plan"] = Field(
        description="The next step in the routing process"
    )


class Cube(BaseModel):
    """Cube data model for robot simulation."""
    name: str
    position: List[float]
    euler: List[float]
    color: str = ""
    
    def __init__(self, **data):
        super().__init__(**data)
        self.color = self._extract_color_from_name(self.name)
    
    def _extract_color_from_name(self, name: str) -> str:
        """Extract color from cube name pattern like cube_blue_03, cube_red_02."""
        parts = name.split('_')
        if len(parts) >= 2 and parts[0] == 'cube':
            return parts[1]
        return "unknown"


class Action(BaseModel):
    """A robot action command."""
    action: Literal["move"] = Field(description="Action type")
    start: str = Field(description="Source cube")
    end: str = Field(description="Target cube")


class Plan(BaseModel):
    """A sequence of robot actions."""
    actions: List[Action] = Field(description="Actions to execute")


class State(TypedDict):
    """State definition for robot task planning graph."""
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    operation: NotRequired[str]
    plan: NotRequired[Plan]


# LLM Initialization
llm = init_chat_model("google_genai:gemini-2.5-flash", max_tokens=10000)
router = llm.with_structured_output(RobotRoute)
planner = llm.with_structured_output(Plan)


# API Helper Functions
def get_cubes_from_api() -> List[Cube]:
    """Get cube information from robot API and return as Cube objects."""
    try:
        response = requests.get(f"{ROBOT_API_BASE_URL}/cube/get", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        cube_data_list = data.get("cube_list", [])
        return [Cube(**cube_data) for cube_data in cube_data_list]
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch cubes from API: {e}")
        raise


def group_cubes_by_color(cubes: List[Cube]) -> dict[str, List[Cube]]:
    """Group cubes by color."""
    color_groups = {}
    for cube in cubes:
        color_groups.setdefault(cube.color, []).append(cube)
    return color_groups


def create_color_summary(color_groups: dict[str, List[Cube]]) -> str:
    """Create color summary string from color groups."""
    return ", ".join([f"{color}: {len(cubes_list)} cubes" for color, cubes_list in color_groups.items()])


def format_cube_info(cubes: List[Cube]) -> str:
    """Format cube information for display."""
    cube_info = []
    for cube in cubes:
        pos = cube.position
        cube_info.append(f"• {cube.name} ({cube.color}): pos({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    return "\n".join(cube_info)


def execute_robot_action(action: Action) -> None:
    """Execute a single robot action through API calls."""
    headers = {"Content-Type": "application/json"}
    
    # Move to start position
    response = requests.post(
        f"{ROBOT_API_BASE_URL}/cmd/move_to",
        json={"target": action.start},
        headers=headers,
        timeout=DEFAULT_TIMEOUT
    )
    response.raise_for_status()
    
    # Pick up object
    response = requests.post(
        f"{ROBOT_API_BASE_URL}/cmd/pick_up",
        json={"target": action.start},
        headers=headers,
        timeout=DEFAULT_TIMEOUT
    )
    response.raise_for_status()
    
    # Move to target position
    response = requests.post(
        f"{ROBOT_API_BASE_URL}/cmd/move_to",
        json={"target": action.end},
        headers=headers,
        timeout=DEFAULT_TIMEOUT
    )
    response.raise_for_status()
    
    # Place object
    response = requests.get(f"{ROBOT_API_BASE_URL}/cmd/place", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()


# Graph Node Functions
def router_node(state: State) -> dict:
    """Routes user requests to appropriate robot operations using AI."""
    try:
        logger.info("Routing robot request")
        decision = router.invoke([
            SystemMessage(content="""Route user requests to Robot operations:

1. 'query_cube' - Get current cube positions and status
2. 'update_cube' - Randomize or modify cube positions  
3. 'plan' - Execute robot movements and actions""")
        ] + state["messages"])
        
        logger.info(f"Robot routing decision: {decision.step}")
        return {"operation": decision.step}
        
    except Exception as e:
        logger.error(f"Error in robot router_node: {e}")
        return {
            "operation": "query_cube",
            "messages": [AIMessage(content=f"Error processing robot request: {e}")]
        }


def route_decision(state: State) -> str:
    """Returns next node to execute based on routing decision."""
    operation = state.get("operation", "query_cube")
    logger.info(f"Routing to robot operation: {operation}")
    return operation


def query_cube_node(state: State) -> dict:
    """Query current cube positions and status."""
    try:
        logger.info("Executing query cube operation")
        cubes = get_cubes_from_api()
        
        if not cubes:
            return {"messages": [AIMessage(content="✅ No cubes found in the environment")]}
        
        color_groups = group_cubes_by_color(cubes)
        color_summary = create_color_summary(color_groups)
        cube_info = format_cube_info(cubes)
        
        result = f"✅ Found {len(cubes)} cubes ({color_summary}):\n{cube_info}"
        return {"messages": [AIMessage(content=result)]}

    except requests.RequestException as e:
        logger.error(f"Robot API connection error: {e}")
        return {"messages": [AIMessage(
            content=f"❌ Failed to connect to robot API: {e}\n"
                   f"(Make sure robot server is running on {ROBOT_API_BASE_URL})"
        )]}
    except Exception as e:
        logger.error(f"Error in query cube: {e}")
        return {"messages": [AIMessage(content=f"❌ Failed to query cube: {e}")]}


def update_cube_node(state: State) -> dict:
    """Update or randomize cube positions."""
    try:
        logger.info("Executing update cube operation")
        response = requests.get(f"{ROBOT_API_BASE_URL}/cube/set", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        
        logger.info("Successfully randomized cube positions")
        return {"messages": [AIMessage(content="✅ Successfully randomized cube positions!")]}

    except requests.RequestException as e:
        logger.error(f"Robot API connection error: {e}")
        return {"messages": [AIMessage(
            content=f"❌ Failed to connect to robot API: {e}\n"
                   f"(Make sure robot server is running on {ROBOT_API_BASE_URL})"
        )]}
    except Exception as e:
        logger.error(f"Error in update cube: {e}")
        return {"messages": [AIMessage(content=f"❌ Failed to update cube: {e}")]}


def plan_node(state: State) -> dict:
    """Execute robot movements and actions."""
    try:
        logger.info("Executing robot planning and execution")
        
        # Extract user instruction
        user_instruction = ""
        if state["messages"]:
            latest_message = state["messages"][-1]
            if hasattr(latest_message, 'content'):
                user_instruction = latest_message.content
        
        cubes = get_cubes_from_api()
        if not cubes:
            return {"messages": [AIMessage(content="❌ No cubes found in environment. Cannot plan execution.")]}
        
        # Create cube analysis
        color_groups = group_cubes_by_color(cubes)
        color_summary = create_color_summary(color_groups)
        cube_state = format_cube_info(cubes)
        
        # Generate plan using LLM
        planning_prompt = f"""You are a household robot command generator. Your task is to analyze object positions and generate optimal movement commands to group objects by color.

COMMAND FORMAT:
{{"action": "move", "start": "cube_name", "end": "target_cube_name"}}

RULES:
1. Analyze the current positions of all objects
2. Group objects by color at the most central or strategic location  
3. Minimize total movement distance
4. Generate commands in logical sequence
5. Only move objects to positions of other objects with the same color

CURRENT OBJECT POSITIONS:
{cube_state}

TASK: {user_instruction}"""
        
        logger.info("Generating structured movement plan with LLM")
        plan_response = planner.invoke([
            SystemMessage(content="You are a robot task planner. Generate a structured plan with specific movement actions."),
            HumanMessage(content=planning_prompt)
        ])
        
        # Format the plan output
        actions_text = []
        for i, action in enumerate(plan_response.actions, 1):
            actions_text.append(f"{i}. {action.action.upper()}: {action.start} → {action.end}")
        
        actions_summary = "\n".join(actions_text) if actions_text else "No actions required - cubes are already optimally positioned"
        
        result = f"""🤖 **Robot Planning Complete**

**Environment Status:**
- {len(cubes)} cubes detected ({color_summary})

**Generated Plan ({len(plan_response.actions)} actions):**
{actions_summary}

**Structured Plan Object:**
```json
{plan_response.model_dump_json(indent=2)}
```

➡️ **Next: Executing plan...**"""
        
        return {
            "messages": [AIMessage(content=result)],
            "plan": plan_response
        }

    except requests.RequestException as e:
        logger.error(f"Robot API connection error: {e}")
        return {"messages": [AIMessage(
            content=f"❌ Failed to connect to robot API: {e}\n"
                   f"(Make sure robot server is running on {ROBOT_API_BASE_URL})"
        )]}
    except Exception as e:
        logger.error(f"Error in plan execution: {e}")
        return {"messages": [AIMessage(content=f"❌ Failed to execute robot planning: {e}")]}


def exec_node(state: State) -> dict:
    """Execute the planned robot actions one by one."""
    try:
        logger.info("Executing planned robot actions")
        
        plan = state.get("plan")
        if not plan or not plan.actions:
            return {"messages": [AIMessage(content="❌ No plan found to execute")]}
        
        execution_results = []
        successful_actions = 0
        failed_actions = 0
        
        # Execute each action
        for i, action in enumerate(plan.actions, 1):
            logger.info(f"Executing action {i}/{len(plan.actions)}: {action.action} {action.start} → {action.end}")
            
            try:
                execute_robot_action(action)
                execution_results.append(f"✅ Action {i}: {action.action.upper()} {action.start} → {action.end}")
                successful_actions += 1
                
            except Exception as action_error:
                logger.error(f"Failed to execute action {i}: {action_error}")
                execution_results.append(f"❌ Action {i}: FAILED - {action.action.upper()} {action.start} → {action.end}")
                failed_actions += 1
        
        # Generate execution summary
        execution_summary = "\n".join(execution_results)
        
        if failed_actions == 0:
            status_emoji = "🎉"
            status_text = "All actions completed successfully!"
        elif successful_actions > 0:
            status_emoji = "⚠️"
            status_text = f"Partial success: {successful_actions} succeeded, {failed_actions} failed"
        else:
            status_emoji = "❌"
            status_text = "All actions failed"
        
        result = f"""{status_emoji} **Robot Execution Complete**

**Execution Summary:**
{status_text}

**Action Results:**
{execution_summary}

**Final Status:**
- Total actions: {len(plan.actions)}
- Successful: {successful_actions}
- Failed: {failed_actions}"""
        
        return {"messages": [AIMessage(content=result)]}
        
    except Exception as e:
        logger.error(f"Error in robot execution: {e}")
        return {"messages": [AIMessage(content=f"❌ Failed to execute robot plan: {e}")]}


# Graph Creation
def create_robot_graph(checkpointer=None):
    """Create robot management graph with optional checkpointer for memory."""
    graph_builder = (
        StateGraph(State)
        .add_node("router", router_node)
        .add_node("query_cube", query_cube_node)
        .add_node("update_cube", update_cube_node)
        .add_node("plan", plan_node)
        .add_node("exec", exec_node)
        .add_edge(START, "router")
        .add_conditional_edges(
            "router",
            route_decision,
            {
                "query_cube": "query_cube",
                "update_cube": "update_cube", 
                "plan": "plan"
            }
        )
        .add_edge("query_cube", END)
        .add_edge("update_cube", END)
        .add_edge("plan", "exec")
        .add_edge("exec", END)
    )
    
    return graph_builder.compile(checkpointer=checkpointer) if checkpointer else graph_builder.compile()


# Default graph instance
graph = create_robot_graph()

__all__ = ["graph", "create_robot_graph"]