"""DynamicSwarm - automatically construct and execute multi-agent workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable

from strands import Agent
from strands.hooks import HookProvider, HookRegistry
from strands.multiagent.base import MultiAgentResult, Status
from strands.multiagent.graph import (
    AfterNodeCallEvent,
    BeforeNodeCallEvent,
    Graph,
    GraphBuilder,
    GraphResult,
)
from strands.session.file_session_manager import FileSessionManager

from .events import (
    AGENT_COLORS,
    ExecutionCompletedEvent,
    ExecutionStartedEvent,
    PlanningStartedEvent,
    PrintingHookProvider,
    SwarmCompletedEvent,
    SwarmFailedEvent,
    SwarmStartedEvent,
    create_colored_callback_handler,
)
from .task import TaskManager

if TYPE_CHECKING:
    from strands.models import Model


@dataclass(frozen=True)
class DynamicSwarmCapabilities:
    """Immutable configuration for available tools and models (created once per swarm)."""

    available_tools: dict[str, Callable[..., Any]]
    available_models: dict[str, "Model"]
    default_model: str | None = None

    def validate_tools(self, tool_names: list[str]) -> None:
        """Validate that all tool names exist in available tools."""
        for tool_name in tool_names:
            if tool_name not in self.available_tools:
                available = list(self.available_tools.keys())
                raise ValueError(
                    f"Tool '{tool_name}' not in available tools: {available}"
                )

    def validate_model(self, model_name: str | None) -> None:
        """Validate that the model name exists in available models."""
        if model_name and model_name not in self.available_models:
            available = list(self.available_models.keys())
            raise ValueError(
                f"Model '{model_name}' not in available models: {available}"
            )

    @property
    def available_tool_names(self) -> list[str]:
        return list(self.available_tools.keys())

    @property
    def available_model_names(self) -> list[str]:
        return list(self.available_models.keys())


class SwarmInstance:
    """Per-query swarm instance holding sub-agents and tasks created during planning."""

    def __init__(
        self,
        capabilities: DynamicSwarmCapabilities,
        hook_registry: HookRegistry | None = None,
    ) -> None:
        self._capabilities = capabilities
        self._hook_registry = hook_registry
        self.sub_agents: dict[str, AgentDefinition] = {}
        self.tasks: dict[str, TaskDefinition] = {}
        self._color_index = 0

    @property
    def capabilities(self) -> DynamicSwarmCapabilities:
        return self._capabilities

    def emit(self, event: Any) -> None:
        """Emit an event to the hook registry."""
        if self._hook_registry and self._hook_registry.has_callbacks():
            self._hook_registry.invoke_callbacks(event)

    def register_agent(self, definition: AgentDefinition) -> None:
        """Register a sub-agent definition."""
        if definition.name in self.sub_agents:
            raise ValueError(f"Agent '{definition.name}' already exists")

        self._capabilities.validate_tools(definition.tools)
        self._capabilities.validate_model(definition.model)

        definition.color = AGENT_COLORS[self._color_index % len(AGENT_COLORS)]
        self._color_index += 1

        self.sub_agents[definition.name] = definition

    def register_task(self, definition: TaskDefinition) -> None:
        """Register a task definition."""
        if definition.name in self.tasks:
            raise ValueError(f"Task '{definition.name}' already exists")

        if definition.agent not in self.sub_agents:
            available = list(self.sub_agents.keys())
            raise ValueError(
                f"Agent '{definition.agent}' not found. Available: {available}"
            )

        for dep in definition.depends_on:
            if dep not in self.tasks:
                available = list(self.tasks.keys())
                raise ValueError(
                    f"Dependency '{dep}' not found. Available: {available}"
                )

        self.tasks[definition.name] = definition

    def get_summary(self) -> str:
        """Get a summary of registered sub-agents and tasks."""
        lines = [
            f"Agents ({len(self.sub_agents)}):",
            *[f"  - {name}: {d.role}" for name, d in self.sub_agents.items()],
            f"\nTasks ({len(self.tasks)}):",
            *[
                f"  - {name} -> {d.agent}" + (f" (depends: {d.depends_on})" if d.depends_on else "")
                for name, d in self.tasks.items()
            ],
        ]
        return "\n".join(lines)


@dataclass
class SessionConfig:
    session_id: str
    storage_dir: str = "./.swarm_sessions"

    def for_agent(self, agent_name: str) -> FileSessionManager:
        return FileSessionManager(
            session_id=f"{self.session_id}-{agent_name}",
            storage_dir=self.storage_dir,
        )

    def for_graph(self) -> FileSessionManager:
        return FileSessionManager(
            session_id=f"{self.session_id}-graph",
            storage_dir=self.storage_dir,
        )


@dataclass
class AgentDefinition:
    name: str
    role: str
    instructions: str | None = None
    tools: list[str] = field(default_factory=list)
    model: str | None = None
    color: str | None = None

    def build_system_prompt(self) -> str:
        parts = [f"You are a {self.role}."]
        if self.instructions:
            parts.append(f"\n\nInstructions:\n{self.instructions}")
        return "\n".join(parts)


@dataclass
class TaskDefinition:
    name: str
    agent: str
    description: str | None = None
    depends_on: list[str] = field(default_factory=list)


class _TaskLifecycleHook(HookProvider):
    """Hook provider that tracks task lifecycle via graph node execution events."""

    def __init__(self, task_manager: TaskManager) -> None:
        self._task_manager = task_manager

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(BeforeNodeCallEvent, self._on_node_start)
        registry.add_callback(AfterNodeCallEvent, self._on_node_complete)

    def _on_node_start(self, event: BeforeNodeCallEvent) -> None:
        task = self._task_manager.get(event.node_id)
        if task and task.is_pending:
            self._task_manager.start(event.node_id)

    def _on_node_complete(self, event: AfterNodeCallEvent) -> None:
        task = self._task_manager.get(event.node_id)
        if task and task.is_executing:
            # AfterNodeCallEvent doesn't carry result/error - just mark as complete
            # The actual result is available in the GraphResult after execution
            self._task_manager.complete(event.node_id)


def build_swarm(
    context: SwarmInstance,
    *,
    use_colored_output: bool = False,
    execution_timeout: float = 900.0,
    task_timeout: float = 300.0,
    session_config: SessionConfig | None = None,
) -> Graph:
    """Build a swarm graph from a SwarmInstance."""
    capabilities = context.capabilities

    if not context.tasks:
        raise ValueError("No tasks registered - cannot build swarm")

    # Build strands Agent instances from definitions
    agents: dict[str, Agent] = {}
    for name, definition in context.sub_agents.items():
        tools = [capabilities.available_tools[t] for t in definition.tools]
        model_name = definition.model or capabilities.default_model
        model = capabilities.available_models.get(model_name) if model_name else None

        callback_handler = None
        if use_colored_output and definition.color:
            callback_handler = create_colored_callback_handler(definition.color, name)

        agents[name] = Agent(
            name=name,
            system_prompt=definition.build_system_prompt(),
            model=model,
            tools=tools or None,  # type: ignore[arg-type]
            callback_handler=callback_handler,
            session_manager=session_config.for_agent(name) if session_config else None,
        )

    # Create TaskManager from definitions (tracks execution state)
    task_manager = TaskManager(context.tasks, hook_registry=context._hook_registry)

    # Build the execution graph
    builder = GraphBuilder()
    for task_name, task in task_manager.all_tasks.items():
        builder.add_node(agents[task.agent], task_name)
        for dep_name in task.depends_on:
            if dep_name in task_manager:
                builder.add_edge(dep_name, task_name)

    builder.set_execution_timeout(execution_timeout)
    builder.set_node_timeout(task_timeout)
    if session_config:
        builder.set_session_manager(session_config.for_graph())
    builder.set_hook_providers([_TaskLifecycleHook(task_manager)])

    return builder.build()


@dataclass
class DynamicSwarmResult:
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


class DynamicSwarm:
    """Dynamically construct and execute multi-agent workflows.

    The orchestrator agent analyzes queries and coordinates multi-agent workflows by:
    1. Planning and creating specialized subagents
    2. Assigning tasks with dependencies
    3. Generating final responses from task outputs

    Subagents run in parallel unless dependencies are declared.
    """

    ORCHESTRATION_PROMPT_TEMPLATE: str = dedent("""\
        Analyze this request and design a workflow:

        {query}

        Available tools: {available_tools}
        Available models: {available_models}

        Create the necessary agents and tasks, then call execute_swarm() when ready.""")

    COMPLETION_PROMPT_TEMPLATE: str = dedent("""\
        The tasks you designed have completed. Here are the outputs from each agent:

        {task_outputs}

        Now synthesize these results into a final, cohesive response to the original query:
        {query}
        
        Be direct and deliver the result. Don't explain the process or mention the sub-agents.""")

    def __init__(
        self,
        available_tools: dict[str, Callable[..., Any]] | None = None,
        available_models: dict[str, "Model"] | None = None,
        *,
        orchestrator_model: "Model" | None = None,
        default_agent_model: str | None = None,
        max_iterations: int = 20,
        execution_timeout: float = 900.0,
        task_timeout: float = 300.0,
        session_id: str | None = None,
        session_storage_dir: str = "./.swarm_sessions",
        hooks: list[HookProvider] | None = None,
        verbose: bool = False,
    ) -> None:
        # Create immutable capabilities once
        self._capabilities = DynamicSwarmCapabilities(
            available_tools=available_tools or {},
            available_models=available_models or {},
            default_model=default_agent_model,
        )
        self._orchestrator_model = orchestrator_model
        self._max_iterations = max_iterations
        self._execution_timeout = execution_timeout
        self._task_timeout = task_timeout

        # Create session config for per-agent memory persistence
        self._session_config = (
            SessionConfig(session_id=session_id, storage_dir=session_storage_dir)
            if session_id
            else None
        )

        self._hook_registry = HookRegistry()

        if verbose:
            self._hook_registry.add_hook(PrintingHookProvider())

        if hooks:
            for hook in hooks:
                self._hook_registry.add_hook(hook)

    def execute(self, query: str) -> DynamicSwarmResult:
        return asyncio.get_event_loop().run_until_complete(self.execute_async(query))

    def _emit(self, event: Any) -> None:
        if self._hook_registry.has_callbacks():
            self._hook_registry.invoke_callbacks(event)

    async def execute_async(self, query: str) -> DynamicSwarmResult:
        # Create fresh execution context per query
        context = SwarmInstance(
            capabilities=self._capabilities,
            hook_registry=self._hook_registry,
        )

        self._emit(SwarmStartedEvent(
            query=query,
            available_tools=self._capabilities.available_tool_names,
            available_models=self._capabilities.available_model_names,
        ))

        # =================================================================
        # Orchestrator Phase 1 & 2: Planning, Creating Subagents, Assigning Tasks
        # =================================================================
        self._emit(PlanningStartedEvent())

        planning_result = await self._run_planning(query, context)
        if not planning_result.success:
            self._emit(SwarmFailedEvent(
                error=planning_result.error or "Orchestration failed"
            ))
            return DynamicSwarmResult(
                status=Status.FAILED,
                planning_output=planning_result.output,
                error=planning_result.error,
            )

        if not context.tasks:
            self._emit(SwarmFailedEvent(
                error="Orchestration completed but no tasks were created"
            ))
            return DynamicSwarmResult(
                status=Status.FAILED,
                planning_output=planning_result.output,
                error="Orchestration completed but no tasks were created",
            )

        graph = build_swarm(
            context,
            use_colored_output=self._hook_registry.has_callbacks(),
            execution_timeout=self._execution_timeout,
            task_timeout=self._task_timeout,
            session_config=self._session_config,
        )

        self._emit(ExecutionStartedEvent(
            tasks=list(context.tasks.keys()),
        ))

        # Graph execution handles task lifecycle via _TaskLifecycleHook
        execution_result = await graph.invoke_async(query)

        self._emit(ExecutionCompletedEvent(
            status=str(execution_result.status) if execution_result else "FAILED",
            agent_count=len(context.sub_agents),
            task_count=len(context.tasks),
        ))

        assert planning_result.orchestrator is not None
        final_response = await self._run_completion(
            query, context, execution_result,
            orchestrator=planning_result.orchestrator
        )

        self._emit(SwarmCompletedEvent())

        return DynamicSwarmResult(
            status=execution_result.status if execution_result else Status.FAILED,
            planning_output=planning_result.output,
            execution_result=execution_result,
            final_response=final_response,
            agents_spawned=len(context.sub_agents),
            tasks_created=len(context.tasks),
        )

    async def _run_planning(
        self, query: str, context: SwarmInstance
    ) -> _PlanningResult:
        from .orchestrator import create_orchestrator_agent

        orchestrator = create_orchestrator_agent(
            context=context,
            model=self._orchestrator_model,
        )

        capabilities = context.capabilities
        orchestration_prompt = self.ORCHESTRATION_PROMPT_TEMPLATE.format(
            query=query,
            available_tools=capabilities.available_tool_names or ["none"],
            available_models=capabilities.available_model_names or ["default only"],
        )

        try:
            result = orchestrator(orchestration_prompt)

            output = None
            if hasattr(result, "message") and result.message:
                content = result.message.get("content", [])
                if content and isinstance(content[0], dict):
                    output = content[0].get("text", "")

            return _PlanningResult(success=True, output=output, orchestrator=orchestrator)

        except Exception as e:
            return _PlanningResult(success=False, error=str(e))

    async def _run_completion(
        self,
        query: str,
        context: SwarmInstance,
        execution_result: MultiAgentResult | None,
        orchestrator: Agent,
    ) -> str | None:
        if not execution_result or not hasattr(execution_result, "results"):
            return None

        # Collect all task outputs
        task_outputs: list[str] = []
        for task_name in context.tasks:
            node_result = execution_result.results.get(task_name)
            if node_result:
                task_outputs.append(f"[{task_name}]:\n{node_result.result}")

        if not task_outputs:
            return None

        completion_prompt = self.COMPLETION_PROMPT_TEMPLATE.format(
            query=query,
            task_outputs="\n\n".join(task_outputs),
        )

        try:
            result = orchestrator(completion_prompt)

            if hasattr(result, "message") and result.message:
                content = result.message.get("content", [])
                if content and isinstance(content[0], dict):
                    return content[0].get("text", "")
            return None
        except Exception:
            return None

@dataclass
class _PlanningResult:
    success: bool
    output: str | None = None
    error: str | None = None
    orchestrator: Agent | None = None  # Preserved for completion phase

