"""Task definition and decorator for task-based workflows."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from strands import Agent

T = TypeVar("T")


@dataclass
class TaskResult(Generic[T]):
    """Result from task execution.

    Attributes:
        task_name: Name of the task that produced this result.
        output: The output value from the task.
        success: Whether the task completed successfully.
        error: Error message if task failed.
        metadata: Additional metadata from execution.
    """

    task_name: str
    output: T | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Return True if task succeeded."""
        return self.success


@dataclass
class Task:
    """A unit of work that can be executed by an agent.

    Tasks represent what needs to be done, decoupled from who does it.
    This allows the same agent to handle multiple tasks, or tasks to
    be reassigned without changing the workflow structure.

    Attributes:
        name: Unique identifier for this task.
        executor: Agent that executes this task.
        description: Human-readable description of what this task does.
        dependencies: Tasks that must complete before this one.
        metadata: Additional task metadata.

    Example:
        research = Task("research", executor=researcher_agent)
        analyze = Task("analyze", executor=analyzer_agent).depends_on(research)
    """

    name: str
    executor: Agent | Callable[..., Any]
    description: str = ""
    dependencies: list[Task] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate task configuration."""
        if not self.name:
            raise ValueError("Task name cannot be empty")

    def __hash__(self) -> int:
        """Hash by task name."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Compare tasks by name."""
        if not isinstance(other, Task):
            return False
        return self.name == other.name

    def __repr__(self) -> str:
        """Return string representation."""
        deps = [d.name for d in self.dependencies]
        return f"Task({self.name!r}, deps={deps})"

    def depends_on(self, *tasks: Task) -> Task:
        """Declare dependencies using fluent API.

        Args:
            *tasks: Tasks that must complete before this one.

        Returns:
            Self for chaining.

        Example:
            analyze = Task("analyze", executor=agent).depends_on(research, validate)
        """
        for task in tasks:
            if task not in self.dependencies:
                self.dependencies.append(task)
        return self


TaskFunc = TypeVar("TaskFunc", bound=Callable[..., Any])


def task(
    executor: Agent | Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    depends_on: list[Callable[..., Any]] | None = None,
    **metadata: Any,
) -> Callable[[TaskFunc], TaskFunc]:
    """Decorator to define a task from a function.

    Args:
        executor: Agent that executes this task.
        name: Task name (defaults to function name).
        description: Task description (defaults to docstring).
        depends_on: List of task functions this depends on.
        **metadata: Additional task metadata.

    Returns:
        Decorator function.

    Example:
        @task(executor=researcher)
        def research_market(ctx: TaskContext) -> str:
            '''Research AI market trends.'''
            return "Research the AI market"

        @task(executor=analyzer, depends_on=[research_market])
        def analyze_data(ctx: TaskContext) -> str:
            '''Analyze the research findings.'''
            return f"Analyze: {ctx.original_input}"
    """

    def decorator(func: TaskFunc) -> TaskFunc:
        task_name = name or func.__name__
        task_description = description or (func.__doc__ or "").strip()

        # Resolve dependencies
        task_dependencies: list[Task] = []
        if depends_on:
            for dep in depends_on:
                if hasattr(dep, "_task"):
                    task_dependencies.append(dep._task)
                else:
                    raise ValueError(f"Dependency {dep.__name__} is not a @task decorated function")

        task_obj = Task(
            name=task_name,
            executor=executor,
            description=task_description,
            dependencies=task_dependencies,
            metadata=metadata,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        wrapper._task = task_obj  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def get_task(func: Callable[..., Any]) -> Task:
    """Extract Task object from a @task decorated function."""
    if not hasattr(func, "_task"):
        raise ValueError(f"{func.__name__} is not a @task decorated function")
    return func._task  # type: ignore[attr-defined]
