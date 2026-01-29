"""TaskSwarm - dynamic task collaboration built on Strands Swarm."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, TYPE_CHECKING

from strands import Agent
from strands.multiagent.swarm import Swarm
from strands.multiagent.base import MultiAgentResult, NodeResult, Status

from .task import Task, TaskResult, get_task
from .context import TaskContext

if TYPE_CHECKING:
    from strands.session import SessionManager
    from strands.hooks import HookProvider


@dataclass
class TaskSwarmResult:
    """Result from TaskSwarm execution."""

    status: Status
    results: dict[str, TaskResult[Any]] = field(default_factory=dict)
    task_history: list[str] = field(default_factory=list)
    execution_time_ms: int = 0

    def get_output(self, task_name: str) -> Any | None:
        """Get output from a specific task."""
        result = self.results.get(task_name)
        return result.output if result and result.success else None

    def __bool__(self) -> bool:
        """Return True if completed successfully."""
        return self.status == Status.COMPLETED


class TaskSwarmBuilder:
    """Builder for constructing TaskSwarm."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._entry_task: str | None = None
        self._max_handoffs: int = 20
        self._max_iterations: int = 20
        self._execution_timeout: float = 900.0
        self._task_timeout: float = 300.0
        self._swarm_id: str = "task_swarm"
        self._session_manager: SessionManager | None = None
        self._hooks: list[HookProvider] | None = None

    def add_task(self, task: Task) -> TaskSwarmBuilder:
        """Add a task to the swarm."""
        if task.name in self._tasks:
            raise ValueError(f"Task '{task.name}' already exists")
        self._tasks[task.name] = task
        return self

    def add_tasks(self, *tasks: Task) -> TaskSwarmBuilder:
        """Add multiple tasks."""
        for task in tasks:
            self.add_task(task)
        return self

    def set_entry_task(self, task: Task | str) -> TaskSwarmBuilder:
        """Set the starting task."""
        self._entry_task = task.name if isinstance(task, Task) else task
        return self

    def set_max_handoffs(self, n: int) -> TaskSwarmBuilder:
        self._max_handoffs = n
        return self

    def set_max_iterations(self, n: int) -> TaskSwarmBuilder:
        self._max_iterations = n
        return self

    def set_execution_timeout(self, seconds: float) -> TaskSwarmBuilder:
        self._execution_timeout = seconds
        return self

    def set_task_timeout(self, seconds: float) -> TaskSwarmBuilder:
        self._task_timeout = seconds
        return self

    def set_swarm_id(self, id: str) -> TaskSwarmBuilder:
        self._swarm_id = id
        return self

    def set_session_manager(self, manager: SessionManager) -> TaskSwarmBuilder:
        self._session_manager = manager
        return self

    def set_hooks(self, hooks: list[HookProvider]) -> TaskSwarmBuilder:
        self._hooks = hooks
        return self

    def build(self) -> TaskSwarm:
        """Build the TaskSwarm."""
        if not self._tasks:
            raise ValueError("Cannot build empty task swarm")
        return TaskSwarm(
            tasks=dict(self._tasks),
            entry_task=self._entry_task,
            max_handoffs=self._max_handoffs,
            max_iterations=self._max_iterations,
            execution_timeout=self._execution_timeout,
            task_timeout=self._task_timeout,
            swarm_id=self._swarm_id,
            session_manager=self._session_manager,
            hooks=self._hooks,
        )


class TaskSwarm:
    """Dynamic task collaboration built on Strands Swarm.

    Tasks can hand off to each other dynamically. Use for adaptive workflows
    where the execution path depends on intermediate results.

    Example:
        research = Task("research", executor=researcher)
        analyze = Task("analyze", executor=analyzer)
        swarm = TaskSwarm([research, analyze])
        result = swarm.execute("Research and analyze AI trends")
    """

    def __init__(
        self,
        tasks: dict[str, Task] | list[Task],
        *,
        entry_task: str | None = None,
        max_handoffs: int = 20,
        max_iterations: int = 20,
        execution_timeout: float = 900.0,
        task_timeout: float = 300.0,
        swarm_id: str = "task_swarm",
        session_manager: SessionManager | None = None,
        hooks: list[HookProvider] | None = None,
    ) -> None:
        if isinstance(tasks, list):
            self._tasks = {t.name: t for t in tasks}
        else:
            self._tasks = tasks

        self._entry_task = entry_task
        self._max_handoffs = max_handoffs
        self._max_iterations = max_iterations
        self._execution_timeout = execution_timeout
        self._task_timeout = task_timeout
        self._swarm_id = swarm_id
        self._session_manager = session_manager
        self._hooks = hooks
        self._swarm = self._build_swarm()

    @classmethod
    def from_decorated(cls, functions: list[Callable[..., Any]], **kwargs: Any) -> TaskSwarm:
        """Create from @task decorated functions."""
        tasks = [get_task(fn) for fn in functions]
        return cls(tasks, **kwargs)

    def _build_swarm(self) -> Swarm:
        """Build underlying Strands Swarm."""
        agents: list[Agent] = []
        entry_agent: Agent | None = None

        for task_name, task in self._tasks.items():
            agent = self._create_agent(task)
            agents.append(agent)
            if self._entry_task and task_name == self._entry_task:
                entry_agent = agent

        return Swarm(
            agents,
            entry_point=entry_agent,
            max_handoffs=self._max_handoffs,
            max_iterations=self._max_iterations,
            execution_timeout=self._execution_timeout,
            node_timeout=self._task_timeout,
            id=self._swarm_id,
            session_manager=self._session_manager,
            hooks=self._hooks,
        )

    def _create_agent(self, task: Task) -> Agent:
        """Create agent wrapper for task."""
        executor = task.executor

        if isinstance(executor, Agent):
            # Build system prompt with task context and available tasks
            base_prompt = executor.system_prompt or ""
            other_tasks = [t for t in self._tasks.keys() if t != task.name]
            
            task_prompt = f"""

[Task: {task.name}]
{task.description or 'Complete the assigned task.'}

Available tasks for handoff: {', '.join(other_tasks) if other_tasks else 'none'}
Use handoff_to_agent tool to delegate to another task if needed."""

            # Create new agent with task name, preserving callback handler
            agent = Agent(
                name=task.name,  # Use task name for handoff
                system_prompt=base_prompt + task_prompt,
                model=executor.model,
                tools=list(executor.tool_registry.registry.values()),
                callback_handler=executor.callback_handler,
            )
            return agent

        elif callable(executor):
            from strands import tool

            @tool
            def execute_task(task_input: str) -> str:
                """Execute the task."""
                ctx = TaskContext(task=task, original_input=task_input)
                return str(executor(ctx))

            other_tasks = [t for t in self._tasks.keys() if t != task.name]
            return Agent(
                name=task.name,
                system_prompt=f"""Task: {task.name}
{task.description or ''}
Use execute_task tool.
Available tasks for handoff: {', '.join(other_tasks) if other_tasks else 'none'}""",
                tools=[execute_task],
            )
        else:
            raise ValueError(f"Executor must be Agent or callable, got {type(executor)}")

    def execute(self, input: str | list[Any]) -> TaskSwarmResult:
        """Execute synchronously."""
        return asyncio.get_event_loop().run_until_complete(self.execute_async(input))

    async def execute_async(self, input: str | list[Any]) -> TaskSwarmResult:
        """Execute asynchronously."""
        result = await self._swarm.invoke_async(input)
        return self._convert_result(result)

    async def stream_async(self, input: str | list[Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream execution events."""
        async for event in self._swarm.stream_async(input):
            if "node_id" in event:
                event["task_name"] = event["node_id"]
            yield event

    def _convert_result(self, result: MultiAgentResult) -> TaskSwarmResult:
        """Convert Strands result to TaskSwarmResult."""
        task_results: dict[str, TaskResult[Any]] = {}
        for node_id, node_result in result.results.items():
            task_results[node_id] = self._convert_node_result(node_id, node_result)

        task_history = []
        if hasattr(result, "node_history"):
            task_history = [n.node_id for n in result.node_history]  # type: ignore

        return TaskSwarmResult(
            status=result.status,
            results=task_results,
            task_history=task_history,
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
    def swarm_id(self) -> str:
        return self._swarm_id
