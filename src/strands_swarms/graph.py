"""TaskGraph - deterministic task pipeline built on Strands Graph."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, TYPE_CHECKING

from strands import Agent
from strands.multiagent.graph import Graph, GraphBuilder
from strands.multiagent.base import MultiAgentResult, NodeResult, Status

from .task import Task, TaskResult, get_task
from .context import TaskContext

if TYPE_CHECKING:
    from strands.session import SessionManager
    from strands.hooks import HookProvider


@dataclass
class TaskGraphResult:
    """Result from TaskGraph execution."""

    status: Status
    results: dict[str, TaskResult[Any]] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    execution_time_ms: int = 0

    def get_output(self, task_name: str) -> Any | None:
        """Get output from a specific task."""
        result = self.results.get(task_name)
        return result.output if result and result.success else None

    def __bool__(self) -> bool:
        """Return True if completed successfully."""
        return self.status == Status.COMPLETED


class TaskGraphBuilder:
    """Builder for constructing TaskGraph."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._entry_points: set[str] = set()
        self._max_executions: int | None = None
        self._execution_timeout: float | None = None
        self._task_timeout: float | None = None
        self._graph_id: str = "task_graph"
        self._session_manager: SessionManager | None = None
        self._hooks: list[HookProvider] | None = None

    def add_task(self, task: Task) -> TaskGraphBuilder:
        """Add a task (dependencies auto-registered)."""
        if task.name in self._tasks:
            raise ValueError(f"Task '{task.name}' already exists")
        self._tasks[task.name] = task
        for dep in task.dependencies:
            if dep.name not in self._tasks:
                self.add_task(dep)
        return self

    def add_tasks(self, *tasks: Task) -> TaskGraphBuilder:
        """Add multiple tasks."""
        for task in tasks:
            self.add_task(task)
        return self

    def set_entry_point(self, task: Task | str) -> TaskGraphBuilder:
        """Set a task as entry point."""
        name = task.name if isinstance(task, Task) else task
        self._entry_points.add(name)
        return self

    def set_max_executions(self, n: int) -> TaskGraphBuilder:
        self._max_executions = n
        return self

    def set_execution_timeout(self, seconds: float) -> TaskGraphBuilder:
        self._execution_timeout = seconds
        return self

    def set_task_timeout(self, seconds: float) -> TaskGraphBuilder:
        self._task_timeout = seconds
        return self

    def set_graph_id(self, id: str) -> TaskGraphBuilder:
        self._graph_id = id
        return self

    def set_session_manager(self, manager: SessionManager) -> TaskGraphBuilder:
        self._session_manager = manager
        return self

    def set_hooks(self, hooks: list[HookProvider]) -> TaskGraphBuilder:
        self._hooks = hooks
        return self

    def build(self) -> TaskGraph:
        """Build the TaskGraph."""
        if not self._tasks:
            raise ValueError("Cannot build empty task graph")
        return TaskGraph(
            tasks=dict(self._tasks),
            entry_points=set(self._entry_points) if self._entry_points else None,
            max_executions=self._max_executions,
            execution_timeout=self._execution_timeout,
            task_timeout=self._task_timeout,
            graph_id=self._graph_id,
            session_manager=self._session_manager,
            hooks=self._hooks,
        )


class TaskGraph:
    """Deterministic task pipeline built on Strands Graph.

    Tasks execute in dependency order. Use for predictable workflows.

    Example:
        research = Task("research", executor=researcher)
        analyze = Task("analyze", executor=analyzer).depends_on(research)
        graph = TaskGraph([research, analyze])
        result = graph.execute("Research AI trends")
    """

    def __init__(
        self,
        tasks: dict[str, Task] | list[Task],
        *,
        entry_points: set[str] | None = None,
        max_executions: int | None = None,
        execution_timeout: float | None = None,
        task_timeout: float | None = None,
        graph_id: str = "task_graph",
        session_manager: SessionManager | None = None,
        hooks: list[HookProvider] | None = None,
    ) -> None:
        if isinstance(tasks, list):
            self._tasks = {t.name: t for t in tasks}
        else:
            self._tasks = tasks

        self._entry_points = entry_points
        self._max_executions = max_executions
        self._execution_timeout = execution_timeout
        self._task_timeout = task_timeout
        self._graph_id = graph_id
        self._session_manager = session_manager
        self._hooks = hooks
        self._graph = self._build_graph()

    @classmethod
    def from_decorated(cls, functions: list[Callable[..., Any]], **kwargs: Any) -> TaskGraph:
        """Create from @task decorated functions."""
        tasks = [get_task(fn) for fn in functions]
        return cls(tasks, **kwargs)

    def _build_graph(self) -> Graph:
        """Build underlying Strands Graph."""
        builder = GraphBuilder()

        for task_name, task in self._tasks.items():
            agent = self._create_agent(task)
            builder.add_node(agent, task_name)

        for task_name, task in self._tasks.items():
            for dep in task.dependencies:
                if dep.name in self._tasks:
                    builder.add_edge(dep.name, task_name)

        if self._entry_points:
            for ep in self._entry_points:
                builder.set_entry_point(ep)

        if self._max_executions:
            builder.set_max_node_executions(self._max_executions)
        if self._execution_timeout:
            builder.set_execution_timeout(self._execution_timeout)
        if self._task_timeout:
            builder.set_node_timeout(self._task_timeout)
        builder.set_graph_id(self._graph_id)
        if self._session_manager:
            builder.set_session_manager(self._session_manager)
        if self._hooks:
            builder.set_hook_providers(self._hooks)

        return builder.build()

    def _create_agent(self, task: Task) -> Agent:
        """Create unique agent wrapper for task."""
        executor = task.executor

        if isinstance(executor, Agent):
            base_prompt = executor.system_prompt or ""
            task_prompt = f"\n\n[Task: {task.name}]\n{task.description or 'Complete the task.'}"
            return Agent(
                name=f"{executor.name or 'agent'}_{task.name}",
                system_prompt=base_prompt + task_prompt,
                model=executor.model,
                tools=list(executor.tool_registry.registry.values()),
                # Preserve callback handler for colored output
                callback_handler=executor.callback_handler,
            )
        elif callable(executor):
            from strands import tool

            @tool
            def execute_task(task_input: str) -> str:
                """Execute the task."""
                ctx = TaskContext(task=task, original_input=task_input)
                return str(executor(ctx))

            return Agent(
                name=f"task_{task.name}",
                system_prompt=f"Task: {task.name}\n{task.description or ''}\nUse execute_task tool.",
                tools=[execute_task],
            )
        else:
            raise ValueError(f"Executor must be Agent or callable, got {type(executor)}")

    def execute(self, input: str | list[Any]) -> TaskGraphResult:
        """Execute synchronously."""
        return asyncio.get_event_loop().run_until_complete(self.execute_async(input))

    async def execute_async(self, input: str | list[Any]) -> TaskGraphResult:
        """Execute asynchronously."""
        result = await self._graph.invoke_async(input)
        return self._convert_result(result)

    async def stream_async(self, input: str | list[Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream execution events."""
        async for event in self._graph.stream_async(input):
            if "node_id" in event:
                event["task_name"] = event["node_id"]
            yield event

    def _convert_result(self, result: MultiAgentResult) -> TaskGraphResult:
        """Convert Strands result to TaskGraphResult."""
        task_results: dict[str, TaskResult[Any]] = {}
        for node_id, node_result in result.results.items():
            task_results[node_id] = self._convert_node_result(node_id, node_result)

        execution_order = []
        if hasattr(result, "execution_order"):
            execution_order = [n.node_id for n in result.execution_order]  # type: ignore

        return TaskGraphResult(
            status=result.status,
            results=task_results,
            execution_order=execution_order,
            total_tasks=len(self._tasks),
            completed_tasks=len([r for r in task_results.values() if r.success]),
            failed_tasks=len([r for r in task_results.values() if not r.success]),
            execution_time_ms=result.execution_time,
        )

    def _convert_node_result(self, task_name: str, node_result: NodeResult) -> TaskResult[Any]:
        """Convert NodeResult to TaskResult."""
        if node_result.status == Status.FAILED:
            return TaskResult(task_name=task_name, success=False, error=str(node_result.result))

        output = None
        if hasattr(node_result.result, "message"):
            msg = node_result.result.message
            if msg and "content" in msg and msg["content"]:
                block = msg["content"][0]
                output = block.get("text") if isinstance(block, dict) else str(block)

        return TaskResult(task_name=task_name, output=output, success=True)

    @property
    def tasks(self) -> dict[str, Task]:
        return self._tasks

    @property
    def graph_id(self) -> str:
        return self._graph_id
