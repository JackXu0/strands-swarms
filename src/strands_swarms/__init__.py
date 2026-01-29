"""Dynamic multi-agent orchestration for Strands Agents.

Given a query, the swarm automatically plans the workflow, spawns specialized
sub-agents, and executes tasks with dependencies.

Current version: Rollout-only (string-in, string-out execution).
RL support coming soon via strands-sglang integration.

Example:
    from strands import tool
    from strands.models import BedrockModel
    from strands_swarms import DynamicSwarm

    @tool
    def search_web(query: str) -> str:
        '''Search the web.'''
        return f"Results for: {query}"

    swarm = DynamicSwarm(
        available_tools={"search_web": search_web},
        available_models={"fast": BedrockModel(model_id="...")},
        verbose=True,
    )
    result = swarm.execute("Research AI trends")
"""

from .dynamic import DynamicSwarm, DynamicSwarmResult
from .events import (
    # Events
    SwarmStartedEvent,
    PlanningStartedEvent,
    AgentSpawnedEvent,
    TaskCreatedEvent,
    PlanningCompletedEvent,
    ExecutionStartedEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    ExecutionCompletedEvent,
    SwarmCompletedEvent,
    SwarmFailedEvent,
    # Hook provider
    PrintingHookProvider,
)

# Re-export strands hook types for convenience
from strands.hooks import HookProvider, HookRegistry

__version__ = "0.1.0"

__all__ = [
    # Main API
    "DynamicSwarm",
    "DynamicSwarmResult",
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
