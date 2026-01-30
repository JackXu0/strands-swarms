"""DynamicSwarm - automatically construct and execute multi-agent workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable

from strands import Agent
from strands.hooks import HookProvider, HookRegistry
from strands.multiagent.base import MultiAgentResult, Status
from strands.multiagent.graph import Graph, GraphBuilder, GraphResult

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
    from strands.session import SessionManager


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


class SwarmConfig:
    def __init__(
        self,
        available_tools: dict[str, Callable[..., Any]],
        available_models: dict[str, Model],
        default_model: str | None = None,
    ) -> None:
        self._available_tools = available_tools
        self._available_models = available_models
        self._default_model = default_model or next(iter(available_models.keys()), None)

        self._agents: dict[str, AgentDefinition] = {}
        self._tasks: dict[str, TaskDefinition] = {}
        self._color_index = 0

    def register_agent(self, definition: AgentDefinition) -> None:
        if definition.name in self._agents:
            raise ValueError(f"Agent '{definition.name}' already exists")

        for tool_name in definition.tools:
            if tool_name not in self._available_tools:
                available = list(self._available_tools.keys())
                raise ValueError(
                    f"Tool '{tool_name}' not in available tools: {available}"
                )

        if definition.model and definition.model not in self._available_models:
            available = list(self._available_models.keys())
            raise ValueError(
                f"Model '{definition.model}' not in available models: {available}"
            )

        definition.color = AGENT_COLORS[self._color_index % len(AGENT_COLORS)]
        self._color_index += 1
        
        self._agents[definition.name] = definition

    def register_task(self, definition: TaskDefinition) -> None:
        if definition.name in self._tasks:
            raise ValueError(f"Task '{definition.name}' already exists")

        if definition.agent not in self._agents:
            available = list(self._agents.keys())
            raise ValueError(
                f"Agent '{definition.agent}' not found. Available: {available}"
            )

        for dep in definition.depends_on:
            if dep not in self._tasks:
                available = list(self._tasks.keys())
                raise ValueError(
                    f"Dependency '{dep}' not found. Available: {available}"
                )

        self._tasks[definition.name] = definition

    def clear(self) -> None:
        self._agents.clear()
        self._tasks.clear()
        self._color_index = 0

    @property
    def agents(self) -> dict[str, AgentDefinition]:
        return dict(self._agents)

    @property
    def tasks(self) -> dict[str, TaskDefinition]:
        return dict(self._tasks)

    @property
    def available_tools(self) -> dict[str, Callable[..., Any]]:
        return self._available_tools

    @property
    def available_models(self) -> dict[str, Model]:
        return self._available_models

    @property
    def default_model(self) -> str | None:
        return self._default_model

    @property
    def available_tool_names(self) -> list[str]:
        return list(self._available_tools.keys())

    @property
    def available_model_names(self) -> list[str]:
        return list(self._available_models.keys())
    
    def get_summary(self) -> str:
        lines = [
            f"Agents ({len(self._agents)}):",
            *[f"  - {name}: {d.role}" for name, d in self._agents.items()],
            f"\nTasks ({len(self._tasks)}):",
            *[
                f"  - {name} -> {d.agent}" + (f" (depends: {d.depends_on})" if d.depends_on else "")
                for name, d in self._tasks.items()
            ],
        ]
        return "\n".join(lines)


@dataclass
class _SwarmBuildResult:
    graph: Graph
    agents: dict[str, Agent]
    task_manager: TaskManager


def build_swarm(
    config: SwarmConfig,
    *,
    use_colored_output: bool = False,
    execution_timeout: float = 900.0,
    task_timeout: float = 300.0,
    session_manager: SessionManager | None = None,
    hook_registry: HookRegistry | None = None,
) -> _SwarmBuildResult:
    # Build all agents first, independent of tasks
    agents: dict[str, Agent] = {}
    
    def build_agent(agent_name: str) -> Agent:
        definition = config.agents.get(agent_name)
        if not definition:
            raise ValueError(f"Agent '{agent_name}' not found")

        tools = [config.available_tools[t] for t in definition.tools]
        model_name = definition.model or config.default_model
        model = config.available_models.get(model_name) if model_name else None

        callback_handler = None
        if use_colored_output and definition.color:
            callback_handler = create_colored_callback_handler(definition.color, agent_name)

        system_prompt = definition.build_system_prompt()

        return Agent(
            name=definition.name,
            system_prompt=system_prompt,
            model=model,
            tools=tools if tools else None,  # type: ignore[arg-type]
            callback_handler=callback_handler,
        )

    # Build each unique agent once
    for agent_name in config.agents:
        agents[agent_name] = build_agent(agent_name)

    if not config.tasks:
        raise ValueError("No tasks registered - cannot build swarm")

    # Create TaskManager and register all tasks
    task_manager = TaskManager(hook_registry=hook_registry)
    
    for task_name, task_def in config.tasks.items():
        task_manager.create(
            name=task_name,
            agent=task_def.agent,
            description=task_def.description,
            depends_on=task_def.depends_on,
        )

    # Build the execution graph using TaskManager data
    builder = GraphBuilder()

    for task_name, task in task_manager.all_tasks.items():
        assigned_agent = agents[task.agent]
        builder.add_node(assigned_agent, task_name)
        
        # Add edges for dependencies
        for dep_name in task.depends_on:
            if dep_name in task_manager:
                builder.add_edge(dep_name, task_name)

    builder.set_execution_timeout(execution_timeout)
    builder.set_node_timeout(task_timeout)

    if session_manager:
        builder.set_session_manager(session_manager)

    return _SwarmBuildResult(
        graph=builder.build(),
        agents=agents,
        task_manager=task_manager,
    )


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
        available_models: dict[str, Model] | None = None,
        *,
        orchestrator_model: Model | None = None,
        default_agent_model: str | None = None,
        max_iterations: int = 20,
        execution_timeout: float = 900.0,
        task_timeout: float = 300.0,
        session_manager: SessionManager | None = None,
        hooks: list[HookProvider] | None = None,
        verbose: bool = False,
    ) -> None:
        self._available_tools = available_tools or {}
        self._available_models = available_models or {}
        self._orchestrator_model = orchestrator_model
        self._default_agent_model = default_agent_model
        self._max_iterations = max_iterations
        self._execution_timeout = execution_timeout
        self._task_timeout = task_timeout
        self._session_manager = session_manager
        
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
        from .orchestrator import set_swarm_config

        config = SwarmConfig(
            available_tools=self._available_tools,
            available_models=self._available_models,
            default_model=self._default_agent_model,
        )

        set_swarm_config(config, hook_registry=self._hook_registry)

        try:
            self._emit(SwarmStartedEvent(
                query=query,
                available_tools=list(self._available_tools.keys()),
                available_models=list(self._available_models.keys()),
            ))
            
            # =================================================================
            # Orchestrator Phase 1 & 2: Planning, Creating Subagents, Assigning Tasks
            # =================================================================
            self._emit(PlanningStartedEvent())
            
            planning_result = await self._run_planning(query, config)
            if not planning_result.success:
                self._emit(SwarmFailedEvent(
                    error=planning_result.error or "Orchestration failed"
                ))
                return DynamicSwarmResult(
                    status=Status.FAILED,
                    planning_output=planning_result.output,
                    error=planning_result.error,
                )

            if not config.tasks:
                self._emit(SwarmFailedEvent(
                    error="Orchestration completed but no tasks were created"
                ))
                return DynamicSwarmResult(
                    status=Status.FAILED,
                    planning_output=planning_result.output,
                    error="Orchestration completed but no tasks were created",
                )

            use_colored_output = self._hook_registry.has_callbacks()
            build_result = build_swarm(
                config,
                use_colored_output=use_colored_output,
                execution_timeout=self._execution_timeout,
                task_timeout=self._task_timeout,
                session_manager=self._session_manager,
                hook_registry=self._hook_registry,
            )

            self._emit(ExecutionStartedEvent(
                tasks=list(config.tasks.keys()),
            ))
            
            # Track task execution with TaskManager
            task_manager = build_result.task_manager
            
            # Start all pending tasks (graph will handle actual execution order)
            for task_name in config.tasks.keys():
                task = task_manager.get(task_name)
                if task and task.is_pending:
                    task_manager.start(task_name)
            
            execution_result = await build_result.graph.invoke_async()
            
            # Update task statuses based on execution results
            if execution_result and hasattr(execution_result, "results"):
                for task_name, node_result in execution_result.results.items():
                    task = task_manager.get(task_name)
                    if task and task.is_executing:
                        if node_result.status == Status.COMPLETED:
                            task_manager.complete(task_name, result=node_result.result)
                        elif node_result.status == Status.FAILED:
                            error_msg = str(node_result.result) if node_result.result else "Task failed"
                            task_manager.fail(task_name, error=error_msg)
                        elif node_result.status == Status.INTERRUPTED:
                            task_manager.interrupt(task_name, reason="Execution interrupted")

            self._emit(ExecutionCompletedEvent(
                status=str(execution_result.status) if execution_result else "FAILED",
                agent_count=len(config.agents),
                task_count=len(config.tasks),
            ))

            assert planning_result.orchestrator is not None
            final_response = await self._run_completion(
                query, config, execution_result, 
                orchestrator=planning_result.orchestrator
            )
            
            self._emit(SwarmCompletedEvent())

            return DynamicSwarmResult(
                status=execution_result.status if execution_result else Status.FAILED,
                planning_output=planning_result.output,
                execution_result=execution_result,
                final_response=final_response,
                agents_spawned=len(config.agents),
                tasks_created=len(config.tasks),
            )

        finally:
            set_swarm_config(None)

    async def _run_planning(
        self, query: str, state: SwarmConfig
    ) -> _PlanningResult:
        from .orchestrator import create_orchestrator_agent

        orchestrator = create_orchestrator_agent(model=self._orchestrator_model)

        orchestration_prompt = self.ORCHESTRATION_PROMPT_TEMPLATE.format(
            query=query,
            available_tools=state.available_tool_names or ['none'],
            available_models=state.available_model_names or ['default only'],
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
        state: SwarmConfig,
        execution_result: MultiAgentResult | None,
        orchestrator: Agent,
    ) -> str | None:
        if not execution_result or not hasattr(execution_result, "results"):
            return None

        # Collect all task outputs
        task_outputs: list[str] = []
        for task_name in state.tasks:
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

