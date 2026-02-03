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

    # Basic usage (stateless agents)
    swarm = DynamicSwarm(
        available_tools={"search_web": search_web},
        verbose=True,
    )
    result = swarm.execute("Research AI trends and summarize")

    # With session persistence (agents maintain memory across tasks)
    swarm_with_memory = DynamicSwarm(
        available_tools={"search_web": search_web},
        session_id="project-alpha",  # Enables per-agent session persistence
        session_storage_dir="./.swarm_sessions",
        verbose=True,
    )
    result = swarm_with_memory.execute("Research AI trends")
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
    TaskInterruptedEvent,
    TaskStartedEvent,
)
from .definition import (
    AgentDefinition,
    DynamicSwarmCapabilities,
    SessionConfig,
    SwarmDefinition,
    TaskDefinition,
)
from .dynamic_swarm import DynamicSwarm, DynamicSwarmResult, build_swarm
from .orchestrator import create_orchestrator_agent
from .task import Task, TaskManager

__version__ = "0.1.1"

__all__ = [
    # Main API
    "DynamicSwarm",
    "DynamicSwarmResult",
    "build_swarm",
    # Definition types
    "DynamicSwarmCapabilities",
    "SwarmDefinition",
    "AgentDefinition",
    "TaskDefinition",
    "SessionConfig",
    # Orchestrator
    "create_orchestrator_agent",
    # Task lifecycle
    "Task",
    "TaskManager",
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
    "TaskInterruptedEvent",
    "ExecutionCompletedEvent",
    "SwarmCompletedEvent",
    "SwarmFailedEvent",
    # Hook system
    "PrintingHookProvider",
    "HookProvider",
    "HookRegistry",
]
