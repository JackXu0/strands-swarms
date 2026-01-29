"""Task execution context for accessing inputs and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .task import Task


@dataclass
class TaskContext:
    """Context passed to task executors during execution.

    Provides access to the original user request and task metadata.

    Attributes:
        task: The task being executed.
        original_input: The original input provided to the graph/swarm.
        metadata: Additional execution metadata.

    Example:
        @task(executor=analyzer)
        def analyze(ctx: TaskContext) -> str:
            return f"Analyzing: {ctx.original_input}"
    """

    task: Task
    original_input: str | list[Any] = ""
    metadata: dict[str, Any] = field(default_factory=dict)
