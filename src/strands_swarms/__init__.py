"""Dynamic multi-agent orchestration for Strands Agents.

This package provides DynamicSwarm - an orchestrator-driven approach to multi-agent
workflows where an LLM automatically designs and executes agent pipelines
based on user queries.

The orchestrator agent handles three responsibilities in a single conversation:
1. Planning and creating subagents - Analyze the task and spawn specialized agents
2. Assigning tasks - Create and assign tasks to the spawned agents
3. Generating final response - Synthesize results into a cohesive response

Using a single orchestrator agent for all three phases (rather than separate agents)
provides better context - the orchestrator knows exactly what it planned and why,
leading to more coherent final responses.

For static multi-agent workflows, use the Strands SDK directly:
- strands.multiagent.swarm.Swarm - dynamic handoffs between agents
- strands.multiagent.graph.Graph - dependency-based execution

Example:
    from strands import tool
    from strands_swarms import DynamicSwarm

    @tool
    def search_web(query: str) -> str:
        '''Search the web.'''
        return f"Results for: {query}"

    swarm = DynamicSwarm(
        available_tools={"search_web": search_web},
        verbose=True,
    )
    result = swarm.execute("Research AI trends and summarize")
"""

# Re-export strands types for convenience
from strands.hooks import HookProvider, HookRegistry
from strands.multiagent.base import Status

from .events import (
    AgentSpawnedEvent,
    ExecutionCompletedEvent,
    # Execution events
    ExecutionStartedEvent,
    PlanningCompletedEvent,
    PlanningStartedEvent,
    # Hook provider
    PrintingHookProvider,
    SwarmCompletedEvent,
    SwarmFailedEvent,
    # Planning/Orchestration events
    SwarmStartedEvent,
    TaskCompletedEvent,
    TaskCreatedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from .orchestrator import create_orchestrator_agent
from .swarm import DynamicSwarm, DynamicSwarmResult

__version__ = "0.1.0"

__all__ = [
    # Main API
    "DynamicSwarm",
    "DynamicSwarmResult",
    # Orchestrator (handles both planning AND completion in same conversation)
    "create_orchestrator_agent",
    # Status enum
    "Status",
    # Events (for custom hooks)
    "SwarmStartedEvent",
    "PlanningStartedEvent",
    "AgentSpawnedEvent",
    "TaskCreatedEvent",
    "PlanningCompletedEvent",
    "ExecutionStartedEvent",
    "TaskStartedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "ExecutionCompletedEvent",
    "SwarmCompletedEvent",
    "SwarmFailedEvent",
    # Hook system
    "PrintingHookProvider",
    "HookProvider",
    "HookRegistry",
]
