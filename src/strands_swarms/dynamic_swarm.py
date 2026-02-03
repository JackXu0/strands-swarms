"""DynamicSwarm - automatically construct and execute multi-agent workflows."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from strands import Agent
from strands.multiagent.base import MultiAgentResult, Status
from strands.multiagent.graph import (
    Graph,
    GraphBuilder,
    GraphResult,
)

from .definition import (
    DynamicSwarmCapabilities,
    SessionConfig,
    SwarmDefinition,
)

if TYPE_CHECKING:
    from strands.models import Model


# =============================================================================
# DynamicSwarm
# =============================================================================

class DynamicSwarm:
    """Dynamically construct and execute multi-agent workflows.

    An orchestrator agent analyzes queries and coordinates workflows by:
    1. Planning: Creating specialized sub-agents with tools
    2. Execution: Running tasks with dependencies in parallel where possible
    3. Synthesis: Combining task outputs into a final response
    """

    def __init__(
        self,
        available_tools: dict[str, Callable[..., Any]] | None = None,
        available_models: dict[str, "Model"] | None = None,
        *,
        orchestrator_model: "Model" | None = None,
        default_agent_model: str | None = None,
        execution_timeout: float = 900.0,
        task_timeout: float = 300.0,
        session_id: str | None = None,
        session_storage_dir: str = "./.swarm_sessions",
    ) -> None:
        self._capabilities = DynamicSwarmCapabilities(
            available_tools=available_tools or {},
            available_models=available_models or {},
            default_model=default_agent_model,
        )
        self._orchestrator_model = orchestrator_model
        self._execution_timeout = execution_timeout
        self._task_timeout = task_timeout
        self._session_config = (
            SessionConfig(session_id=session_id, storage_dir=session_storage_dir)
            if session_id
            else None
        )

    def execute(self, query: str) -> DynamicSwarmResult:
        """Execute a query synchronously."""
        return asyncio.run(self.execute_async(query))

    async def execute_async(self, query: str) -> DynamicSwarmResult:
        """Execute a query asynchronously (non-streaming)."""
        result: DynamicSwarmResult | None = None
        async for event in self.stream_async(query):
            if event.get("type") == "swarm_result":
                result = event.get("result")
        if not isinstance(result, DynamicSwarmResult):
            raise RuntimeError("DynamicSwarm stream ended without producing a result")
        return result

    async def stream_async(self, query: str) -> AsyncIterator[dict[str, Any]]:
        """Stream execution events, including planning and graph execution.

        This yields:
        - High-level DynamicSwarm events (planning/execution/synthesis)
        - Raw `strands` Graph stream events (e.g. multiagent_node_start/stop, multiagent_result)
        """
        definition = SwarmDefinition(capabilities=self._capabilities)

        yield {
            "type": "swarm_started",
            "query": query,
            "available_tools": self._capabilities.available_tool_names,
            "available_models": self._capabilities.available_model_names,
        }

        # Phase 1: Planning
        yield {"type": "planning_started"}
        planning_result = await self._run_planning(query, definition)

        if not planning_result.success:
            error = planning_result.error or "Planning failed"
            yield {"type": "planning_failed", "error": error}
            yield {
                "type": "swarm_result",
                "result": DynamicSwarmResult(
                    status=Status.FAILED,
                    planning_output=planning_result.output,
                    agents_spawned=len(definition.sub_agents),
                    tasks_created=len(definition.tasks),
                    error=error,
                ),
            }
            return

        if not definition.tasks:
            error = "No tasks were created"
            yield {"type": "planning_failed", "error": error}
            yield {
                "type": "swarm_result",
                "result": DynamicSwarmResult(
                    status=Status.FAILED,
                    planning_output=planning_result.output,
                    agents_spawned=len(definition.sub_agents),
                    tasks_created=0,
                    error=error,
                ),
            }
            return

        yield {
            "type": "planning_completed",
            "planning_output": planning_result.output,
            "summary": definition.get_summary(),
            "agents": definition.sub_agents,
            "tasks": definition.tasks,
        }

        # Phase 2: Execution
        graph = build_swarm(
            definition,
            execution_timeout=self._execution_timeout,
            task_timeout=self._task_timeout,
            session_config=self._session_config,
        )
        yield {"type": "execution_started", "tasks": list(definition.tasks.keys())}

        execution_result: GraphResult | None = None
        execution_error: str | None = None
        try:
            async for event in graph.stream_async(query):
                if event.get("type") == "multiagent_result":
                    candidate = event.get("result")
                    if isinstance(candidate, GraphResult):
                        execution_result = candidate
                yield event
        except Exception as e:
            execution_error = str(e)

        if execution_error:
            yield {"type": "execution_failed", "error": execution_error}
            yield {
                "type": "swarm_result",
                "result": DynamicSwarmResult(
                    status=Status.FAILED,
                    planning_output=planning_result.output,
                    execution_result=execution_result,
                    final_response=None,
                    agents_spawned=len(definition.sub_agents),
                    tasks_created=len(definition.tasks),
                    error=execution_error,
                ),
            }
            return

        if not execution_result:
            error = "Graph completed without producing a result"
            yield {"type": "execution_failed", "error": error}
            yield {
                "type": "swarm_result",
                "result": DynamicSwarmResult(
                    status=Status.FAILED,
                    planning_output=planning_result.output,
                    execution_result=None,
                    final_response=None,
                    agents_spawned=len(definition.sub_agents),
                    tasks_created=len(definition.tasks),
                    error=error,
                ),
            }
            return

        yield {"type": "execution_completed", "status": execution_result.status.value}

        # Phase 3: Synthesis
        yield {"type": "synthesis_started"}
        assert planning_result.orchestrator is not None
        final_response = await self._synthesize_final_response(
            query, definition, execution_result, planning_result.orchestrator
        )
        yield {"type": "synthesis_completed", "final_response": final_response}

        result = DynamicSwarmResult(
            status=execution_result.status,
            planning_output=planning_result.output,
            execution_result=execution_result,
            final_response=final_response,
            agents_spawned=len(definition.sub_agents),
            tasks_created=len(definition.tasks),
            error=(
                None
                if execution_result.status == Status.COMPLETED
                else f"Execution ended with status {execution_result.status.value}"
            ),
        )

        yield {"type": "swarm_result", "result": result}

    async def _run_planning(
        self, query: str, definition: SwarmDefinition
    ) -> _PlanningResult:
        from .orchestrator import create_orchestrator_agent

        orchestrator = create_orchestrator_agent(
            definition=definition,
            model=self._orchestrator_model,
        )

        prompt = PLANNING_PROMPT.format(
            query=query,
            available_tools=definition.capabilities.available_tool_names or ["none"],
            available_models=definition.capabilities.available_model_names or ["default"],
        )

        try:
            output = _extract_message_text(orchestrator(prompt))
            return _PlanningResult(success=True, output=output, orchestrator=orchestrator)
        except Exception as e:
            return _PlanningResult(success=False, error=str(e))

    async def _synthesize_final_response(
        self,
        query: str,
        definition: SwarmDefinition,
        execution_result: MultiAgentResult | None,
        orchestrator: Agent,
    ) -> str | None:
        if not execution_result or not hasattr(execution_result, "results"):
            return None

        task_outputs: list[str] = []
        for task_name in definition.tasks:
            node_result = execution_result.results.get(task_name)
            if node_result:
                task_outputs.append(f"[{task_name}]:\n{node_result.result}")

        if not task_outputs:
            return None

        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            task_outputs="\n\n".join(task_outputs),
        )

        try:
            return _extract_message_text(orchestrator(prompt))
        except Exception:
            return None


# =============================================================================
# Results
# =============================================================================


@dataclass
class DynamicSwarmResult:
    """Result from DynamicSwarm execution."""

    status: Status
    planning_output: str | None = None
    execution_result: GraphResult | None = None
    final_response: str | None = None
    agents_spawned: int = 0
    tasks_created: int = 0
    error: str | None = None

    def get_output(self, task_name: str) -> Any | None:
        if self.execution_result and hasattr(self.execution_result, "results"):
            node_result = self.execution_result.results.get(task_name)
            if node_result:
                return str(node_result.result)
        return None

    def __bool__(self) -> bool:
        return self.status == Status.COMPLETED


@dataclass
class _PlanningResult:
    success: bool
    output: str | None = None
    error: str | None = None
    orchestrator: Agent | None = None


# =============================================================================
# Graph Building
# =============================================================================

def build_swarm(
    definition: SwarmDefinition,
    *,
    execution_timeout: float = 900.0,
    task_timeout: float = 300.0,
    session_config: SessionConfig | None = None,
) -> Graph:
    """Build a strands Graph from a SwarmDefinition."""
    if not definition.tasks:
        raise ValueError("No tasks registered - cannot build swarm")

    capabilities = definition.capabilities

    # Build strands Agent instances from definitions
    agents: dict[str, Agent] = {}
    for name, agent_def in definition.sub_agents.items():
        tools = [capabilities.available_tools[t] for t in agent_def.tools]
        model_name = agent_def.model or capabilities.default_model
        model = capabilities.available_models.get(model_name) if model_name else None

        agents[name] = Agent(
            name=name,
            system_prompt=agent_def.build_system_prompt(),
            model=model,
            tools=tools or None,  # type: ignore[arg-type]
        )

    # Build the execution graph
    builder = GraphBuilder()
    for task_name, task in definition.tasks.items():
        builder.add_node(agents[task.agent], task_name)
    for task_name, task in definition.tasks.items():
        for dep_name in task.depends_on:
            builder.add_edge(dep_name, task_name)

    builder.set_execution_timeout(execution_timeout)
    builder.set_node_timeout(task_timeout)
    if session_config:
        builder.set_session_manager(session_config.for_graph())

    return builder.build()


# =============================================================================
# Prompts
# =============================================================================


PLANNING_PROMPT = """\
Analyze this request and design a multi-agent workflow to complete it:

REQUEST: {query}

AVAILABLE TOOLS: {available_tools}
AVAILABLE MODELS: {available_models}

INSTRUCTIONS:
1. Break down the request into logical steps
2. Create specialized agents with appropriate tools using spawn_agent()
3. Create tasks with dependencies using create_task()
   - Important: dependencies must already exist (create dependency tasks first)
4. Call finalize_plan() when done

Keep the workflow simple - only create agents and tasks that are necessary."""

SYNTHESIS_PROMPT = """\
The workflow has completed. Here are the results from each task:

{task_outputs}

ORIGINAL REQUEST: {query}

Synthesize these results into a final response. \
Be direct - deliver the answer without mentioning the workflow or agents."""


# =============================================================================
# Helpers
# =============================================================================


def _extract_message_text(result: Any) -> str | None:
    """Extract the first text block from a strands Agent result, if present."""
    message = getattr(result, "message", None)
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    if not isinstance(content, list) or not content:
        return None

    first = content[0]
    if not isinstance(first, dict):
        return None

    text = first.get("text")
    return text if isinstance(text, str) else None
