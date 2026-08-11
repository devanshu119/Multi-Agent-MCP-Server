"""Define a custom multi-agent workflow for implementing coding solutions."""
import asyncio
import logging
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import State
from react_agent.tools import TOOLS, initialize_tools
from react_agent.utils import load_chat_model
from react_agent.agents.orchestrator import get_orchestrator
from react_agent.agents.planner import get_planner
from react_agent.agents.coder import get_coder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoderWorkflow:
    def __init__(self):
        # Initialize MCP tools and shared model
        self.config = Configuration.load_from_langgraph_json()
        asyncio.run(initialize_tools(self.config))

        # Create shared model instance
        self.llm = load_chat_model(
            self.config.model,
            self.config.openrouter_base_url
        )

        # Initialize agents
        self.orchestrator = get_orchestrator(self.llm, TOOLS)
        self.planner = get_planner(self.llm, TOOLS)
        self.coder = get_coder(self.llm, TOOLS)

    def route_agent(self, state: State) -> Literal["MCP", "__end__"]:
        """Route next steps based on tool calls."""
        last_message = state.messages[-1]
        
        # Check for tool calls natively
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info(f"Found tool calls - Routing to MCP from {state.current_agent}")
            return "MCP"
        
        logger.info(f"No tool calls - Ending workflow from {state.current_agent}")
        return "__end__"

    def setup_workflow(self):
        """Set up the workflow graph."""
        workflow = StateGraph(State)

        # Add nodes for each agent
        workflow.add_node("orchestrator", self.orchestrator.run)
        workflow.add_node("planner", self.planner.run)
        workflow.add_node("coder", self.coder.run)
        
        # Native ToolNode from LangGraph 0.3
        workflow.add_node("MCP", ToolNode(TOOLS))

        # Set orchestrator as the entrypoint
        workflow.add_edge("__start__", "orchestrator")

        # Add conditional edges for routing from agents
        workflow.add_conditional_edges("orchestrator", self.route_agent, {"MCP": "MCP", "__end__": END})
        workflow.add_conditional_edges("planner", self.route_agent, {"MCP": "MCP", "__end__": END})
        workflow.add_conditional_edges("coder", self.route_agent, {"MCP": "MCP", "__end__": END})

        # Add conditional edges from MCP back to agents
        def route_mcp(state: State) -> str:
            """Route back to the calling agent.
            Commands returned by routing tools will automatically update current_agent.
            """
            return state.current_agent or 'orchestrator'

        workflow.add_conditional_edges(
            "MCP",
            route_mcp,
            {
                "orchestrator": "orchestrator",
                "planner": "planner",
                "coder": "coder"
            }
        )

        return workflow.compile()

    async def execute(self, task: str):
        """Execute the workflow."""
        logger.info("Initiating workflow...")
        workflow = self.setup_workflow()

        logger.info(f"Initial task: {task}")

        # Create proper initial state with HumanMessage
        initial_state = State(
            messages=[HumanMessage(content=task)],
            current_agent="orchestrator"
        )

        config = {"recursion_limit": 50}
        async for output in workflow.astream(initial_state, stream_mode="updates", config=config):
            logger.info(f"Agent message: {str(output)}")

# For LangGraph Studio support
coder_workflow = CoderWorkflow()
graph = coder_workflow.setup_workflow()
