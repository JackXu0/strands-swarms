"""DynamicSwarm - automatically construct and execute multi-agent workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from strands.multiagent.base import Status
from strands.hooks import HookProvider, HookRegistry

from .registry import DynamicRegistry
from .planner import set_active_registry, create_planner_agent
from .task import Task
from .swarm import TaskSwarm, TaskSwarmResult
from .graph import TaskGraph, TaskGraphResult
from .events import (
    SwarmStartedEvent,
    PlanningStartedEvent,
    PlanningCompletedEvent,
    ExecutionStartedEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
    ExecutionCompletedEvent,
    SwarmCompletedEvent,
    SwarmFailedEvent,
    PrintingHookProvider,
)

if TYPE_CHECKING:
    from strands.models import Model
    from strands.session import SessionManager


@dataclass
class DynamicSwarmResult:
    """Result from DynamicSwarm execution.

    Attributes:
        status: Overall execution status.
        planning_output: Output from the planning phase.
        execution_result: Result from the execution phase (TaskSwarmResult or TaskGraphResult).
        final_response: Final response from the planner after all tasks completed.
        agents_spawned: Number of agents that were dynamically spawned.
        tasks_created: Number of tasks that were created.
        execution_mode: The mode used for execution (swarm or graph).
    """

    status: Status
    planning_output: str | None = None
    execution_result: TaskSwarmResult | TaskGraphResult | None = None
    final_response: str | None = None
    agents_spawned: int = 0
    tasks_created: int = 0
    execution_mode: str = "swarm"
    error: str | None = None

    def get_output(self, task_name: str) -> Any | None:
        """Get output from a specific task."""
        if self.execution_result:
            return self.execution_result.get_output(task_name)
        return None

    def __bool__(self) -> bool:
        """Return True if completed successfully."""
        return self.status == Status.COMPLETED


class DynamicSwarm:
    """Dynamically construct and execute multi-agent workflows.

    DynamicSwarm uses a planner agent to analyze user queries and automatically
    spawn specialized sub-agents, create tasks, and execute the workflow.

    The planner can:
    - Spawn agents with specific roles, tools, and models
    - Create tasks with dependencies
    - Choose between swarm (dynamic) and graph (deterministic) execution

    Note:
        Current version supports rollout execution only (string-in, string-out).
        RL support is planned via strands-sglang integration.

    Example:
        from strands_swarms import DynamicSwarm
        from strands import tool

        @tool
        def search_web(query: str) -> str:
            '''Search the web.'''
            return f"Results for: {query}"

        @tool
        def write_file(path: str, content: str) -> str:
            '''Write to a file.'''
            return f"Wrote to {path}"

        swarm = DynamicSwarm(
            available_tools={
                "search_web": search_web,
                "write_file": write_file,
            },
            available_models={
                "powerful": "anthropic.claude-sonnet-4-20250514",
                "fast": "anthropic.claude-haiku-3-20250722",
            },
        )

        result = swarm.execute("Research AI trends and write a summary")
    """

    def __init__(
        self,
        available_tools: dict[str, Callable[..., Any]] | None = None,
        available_models: dict[str, Model] | None = None,
        *,
        planner_model: Model | None = None,
        default_agent_model: str | None = None,
        default_mode: str = "swarm",
        max_handoffs: int = 20,
        max_iterations: int = 20,
        execution_timeout: float = 900.0,
        task_timeout: float = 300.0,
        session_manager: SessionManager | None = None,
        hooks: list[HookProvider] | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize DynamicSwarm.

        Args:
            available_tools: Pool of tools that can be assigned to spawned agents.
                            Keys are tool names used by the planner.
            available_models: Pool of Model instances that can be used by spawned agents.
                             Keys are friendly names (e.g., "fast", "powerful"),
                             values are Model instances.
            planner_model: Model instance to use for the planner agent.
            default_agent_model: Default model name (key in available_models) for spawned agents.
            default_mode: Default execution mode ("swarm" or "graph").
            max_handoffs: Maximum handoffs in swarm mode.
            max_iterations: Maximum iterations for execution.
            execution_timeout: Overall execution timeout in seconds.
            task_timeout: Per-task timeout in seconds.
            session_manager: Optional session manager for persistence.
            hooks: List of HookProvider instances for event callbacks.
                  Use PrintingHookProvider() for CLI output.
            verbose: Shorthand for hooks=[PrintingHookProvider()].
        """
        self._available_tools = available_tools or {}
        self._available_models = available_models or {}
        self._planner_model = planner_model
        self._default_agent_model = default_agent_model
        self._default_mode = default_mode
        self._max_handoffs = max_handoffs
        self._max_iterations = max_iterations
        self._execution_timeout = execution_timeout
        self._task_timeout = task_timeout
        self._session_manager = session_manager
        
        # Build hook registry
        self._hook_registry = HookRegistry()
        
        # Add verbose hook if requested
        if verbose:
            self._hook_registry.add_hook(PrintingHookProvider())
        
        # Add user-provided hooks
        if hooks:
            for hook in hooks:
                self._hook_registry.add_hook(hook)

    def execute(self, query: str) -> DynamicSwarmResult:
        """Execute a query by dynamically building and running a workflow.

        Args:
            query: The user's request to process.

        Returns:
            DynamicSwarmResult containing planning and execution results.
        """
        return asyncio.get_event_loop().run_until_complete(self.execute_async(query))

    def _emit(self, event: Any) -> None:
        """Emit an event to all registered hooks."""
        if self._hook_registry.has_callbacks():
            self._hook_registry.invoke_callbacks(event)

    async def execute_async(self, query: str) -> DynamicSwarmResult:
        """Execute asynchronously.

        Args:
            query: The user's request to process.

        Returns:
            DynamicSwarmResult containing planning and execution results.
        """
        # Create registry for this execution
        registry = DynamicRegistry(
            available_tools=self._available_tools,
            available_models=self._available_models,
            default_model=self._default_agent_model,
        )

        # Set registry for planning tools (with hook registry)
        set_active_registry(registry, hook_registry=self._hook_registry)

        try:
            # Emit swarm started event
            self._emit(SwarmStartedEvent(
                query=query,
                available_tools=list(self._available_tools.keys()),
                available_models=list(self._available_models.keys()),
            ))
            
            # Phase 1: Planning
            self._emit(PlanningStartedEvent())
            
            planning_result = await self._run_planning(query, registry)
            if not planning_result.success:
                self._emit(SwarmFailedEvent(
                    error=planning_result.error or "Planning failed"
                ))
                return DynamicSwarmResult(
                    status=Status.FAILED,
                    planning_output=planning_result.output,
                    error=planning_result.error,
                )

            # Validate we have tasks
            if not registry.tasks:
                self._emit(SwarmFailedEvent(
                    error="Planning completed but no tasks were created"
                ))
                return DynamicSwarmResult(
                    status=Status.FAILED,
                    planning_output=planning_result.output,
                    error="Planning completed but no tasks were created",
                )

            # Phase 2: Build and execute workflow
            self._emit(ExecutionStartedEvent(
                mode=registry.execution_mode,
                tasks=list(registry.tasks.keys()),
            ))
            
            execution_result = await self._run_execution(query, registry)

            # Emit execution completion event
            self._emit(ExecutionCompletedEvent(
                status=str(execution_result.status) if execution_result else "FAILED",
                agent_count=len(registry.agents),
                task_count=len(registry.tasks),
            ))

            # Phase 3: Generate final response
            final_response = await self._run_completion(query, registry, execution_result)
            
            self._emit(SwarmCompletedEvent())

            return DynamicSwarmResult(
                status=execution_result.status if execution_result else Status.FAILED,
                planning_output=planning_result.output,
                execution_result=execution_result,
                final_response=final_response,
                agents_spawned=len(registry.agents),
                tasks_created=len(registry.tasks),
                execution_mode=registry.execution_mode,
            )

        finally:
            # Clear registry
            set_active_registry(None)

    async def _run_planning(
        self, query: str, registry: DynamicRegistry
    ) -> _PlanningResult:
        """Run the planning phase.

        Args:
            query: The user's query.
            registry: The registry to populate.

        Returns:
            Planning result with success status and output.
        """
        planner = create_planner_agent(model=self._planner_model)

        # Build planning prompt
        planning_prompt = f"""Analyze this request and design a workflow:

{query}

Available tools: {registry.available_tool_names or ['none']}
Available models: {registry.available_model_names or ['default only']}

Create the necessary agents and tasks, then call execute_swarm() when ready."""

        try:
            # Agent.__call__ is synchronous
            result = planner(planning_prompt)

            # Extract text output
            output = None
            if hasattr(result, "message") and result.message:
                content = result.message.get("content", [])
                if content and isinstance(content[0], dict):
                    output = content[0].get("text", "")

            return _PlanningResult(success=True, output=output)

        except Exception as e:
            return _PlanningResult(success=False, error=str(e))

    async def _run_completion(
        self,
        query: str,
        registry: DynamicRegistry,
        execution_result: TaskSwarmResult | TaskGraphResult | None,
    ) -> str | None:
        """Run the completion phase - generate final response from task outputs.

        Args:
            query: The original user query.
            registry: The registry with agent/task info.
            execution_result: Results from the execution phase.

        Returns:
            Final response string, or None if completion fails.
        """
        if not execution_result:
            return None

        # Collect all task outputs
        task_outputs: list[str] = []
        for task_name in registry.tasks:
            output = execution_result.get_output(task_name)
            if output:
                task_outputs.append(f"[{task_name}]:\n{output}")

        if not task_outputs:
            return None

        # Create completion agent (reuse planner model)
        from strands import Agent
        
        completion_agent = Agent(
            name="completion",
            system_prompt="""You are a completion agent. Your job is to take the outputs from multiple sub-agents and generate a final, cohesive response that answers the user's original query.

Be direct and provide the final answer. Do not explain the process or mention the sub-agents - just deliver the result as if you completed the task yourself.""",
            model=self._planner_model,
        )

        completion_prompt = f"""Original query: {query}

The following outputs were produced by specialized agents:

{chr(10).join(task_outputs)}

Based on these outputs, generate the final response to the original query."""

        try:
            result = completion_agent(completion_prompt)
            
            # Extract text output
            if hasattr(result, "message") and result.message:
                content = result.message.get("content", [])
                if content and isinstance(content[0], dict):
                    return content[0].get("text", "")
            return None
        except Exception:
            return None

    async def _run_execution(
        self, query: str, registry: DynamicRegistry
    ) -> TaskSwarmResult | TaskGraphResult | None:
        """Run the execution phase.

        Args:
            query: The original query.
            registry: The populated registry.

        Returns:
            Execution result from TaskSwarm or TaskGraph.
        """
        # Build tasks from registry
        tasks = self._build_tasks(registry)

        # Determine execution mode
        mode = registry.execution_mode or self._default_mode

        if mode == "graph":
            return await self._execute_graph(query, tasks, registry)
        else:
            return await self._execute_swarm(query, tasks, registry)

    def _build_tasks(self, registry: DynamicRegistry) -> dict[str, Task]:
        """Build Task objects from registry definitions.

        Args:
            registry: The populated registry.

        Returns:
            Dictionary of task name to Task objects.
        """
        tasks: dict[str, Task] = {}

        # First pass: create tasks without dependencies
        for task_name, task_def in registry.tasks.items():
            agent = registry.build_agent(task_def.agent)
            tasks[task_name] = Task(
                name=task_name,
                executor=agent,
                description=task_def.description or "",
            )

        # Second pass: add dependencies
        for task_name, task_def in registry.tasks.items():
            for dep_name in task_def.depends_on:
                if dep_name in tasks:
                    tasks[task_name].dependencies.append(tasks[dep_name])

        return tasks

    async def _execute_swarm(
        self, query: str, tasks: dict[str, Task], registry: DynamicRegistry
    ) -> TaskSwarmResult:
        """Execute using TaskSwarm.

        Args:
            query: The original query.
            tasks: Built Task objects.
            registry: The registry with configuration.

        Returns:
            TaskSwarmResult from execution.
        """
        swarm = TaskSwarm(
            tasks=tasks,
            entry_task=registry.entry_task,
            max_handoffs=self._max_handoffs,
            max_iterations=self._max_iterations,
            execution_timeout=self._execution_timeout,
            task_timeout=self._task_timeout,
            session_manager=self._session_manager,
        )

        if self._hook_registry.has_callbacks():
            # Stream execution to emit task events
            result = None
            current_task = None
            async for event in swarm.stream_async(query):
                task_name = event.get("task_name") or event.get("node_id")
                if task_name and task_name != current_task:
                    if current_task:
                        self._emit(TaskCompletedEvent(name=current_task))
                    agent_def = registry.agents.get(task_name)
                    self._emit(TaskStartedEvent(
                        name=task_name,
                        agent_role=agent_def.role if agent_def else None,
                    ))
                    current_task = task_name
                if "result" in event:
                    result = event["result"]
            if current_task:
                self._emit(TaskCompletedEvent(name=current_task))
            # Get final result
            return swarm._convert_result(result) if result else await swarm.execute_async(query)
        else:
            return await swarm.execute_async(query)

    async def _execute_graph(
        self, query: str, tasks: dict[str, Task], registry: DynamicRegistry
    ) -> TaskGraphResult:
        """Execute using TaskGraph.

        Args:
            query: The original query.
            tasks: Built Task objects.
            registry: The registry with configuration.

        Returns:
            TaskGraphResult from execution.
        """
        entry_points = {registry.entry_task} if registry.entry_task else None

        graph = TaskGraph(
            tasks=tasks,
            entry_points=entry_points,
            execution_timeout=self._execution_timeout,
            task_timeout=self._task_timeout,
            session_manager=self._session_manager,
        )

        if self._hook_registry.has_callbacks():
            # Stream execution to emit task events
            result = None
            current_task = None
            async for event in graph.stream_async(query):
                task_name = event.get("task_name") or event.get("node_id")
                if task_name and task_name != current_task:
                    if current_task:
                        self._emit(TaskCompletedEvent(name=current_task))
                    agent_def = registry.agents.get(task_name)
                    self._emit(TaskStartedEvent(
                        name=task_name,
                        agent_role=agent_def.role if agent_def else None,
                    ))
                    current_task = task_name
                if "result" in event:
                    result = event["result"]
            if current_task:
                self._emit(TaskCompletedEvent(name=current_task))
            # Get final result
            return graph._convert_result(result) if result else await graph.execute_async(query)
        else:
            return await graph.execute_async(query)


@dataclass
class _PlanningResult:
    """Internal result from planning phase."""

    success: bool
    output: str | None = None
    error: str | None = None
