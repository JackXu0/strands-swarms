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

    # Basic usage
    swarm = DynamicSwarm(available_tools={"search_web": search_web})
    result = swarm.execute("Research AI trends and summarize")

    # Streaming trajectory
    import asyncio

    async def run():
        async for event in swarm.stream_async(
            "Research AI trends and summarize",
            include_subagent_events=False,
        ):
            print(event)

    asyncio.run(run())
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from strands.multiagent.base import Status

from .definition import (
    AgentDefinition,
    DynamicSwarmCapabilities,
    SessionConfig,
    SwarmDefinition,
    TaskDefinition,
)
from .dynamic_swarm import DynamicSwarm, DynamicSwarmResult, build_swarm
from .orchestrator import create_orchestrator_agent

try:
    __version__ = _version("strands-swarms")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

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
    "Status",
]
