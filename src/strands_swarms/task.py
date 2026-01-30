"""Task lifecycle management for DynamicSwarm."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from strands.multiagent.base import Status

from .events import (
    TaskCompletedEvent,
    TaskCreatedEvent,
    TaskFailedEvent,
    TaskInterruptedEvent,
    TaskStartedEvent,
)

if TYPE_CHECKING:
    from strands.hooks.registry import HookRegistry


TERMINAL_STATUSES: set[Status] = {Status.COMPLETED, Status.FAILED, Status.INTERRUPTED}


@dataclass
class Task:
    name: str
    agent: str
    description: str | None = None
    depends_on: list[str] = field(default_factory=list)
    status: Status = Status.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None

    _VALID_TRANSITIONS: ClassVar[dict[Status, set[Status]]] = {
        Status.PENDING: {Status.EXECUTING, Status.INTERRUPTED},
        Status.EXECUTING: {Status.COMPLETED, Status.FAILED, Status.INTERRUPTED},
    }

    def transition_to(self, new_status: Status, **kwargs: Any) -> Status:
        valid_next = self._VALID_TRANSITIONS.get(self.status, set())

        if new_status not in valid_next:
            raise ValueError(
                f"Invalid task transition: {self.status.value} -> {new_status.value}. "
                f"Valid transitions from {self.status.value}: {[s.value for s in valid_next]}"
            )

        old_status = self.status
        self.status = new_status

        now = datetime.now()
        if new_status == Status.EXECUTING:
            self.started_at = now
        elif new_status in TERMINAL_STATUSES:
            self.completed_at = now

        if "result" in kwargs:
            self.result = kwargs["result"]
        if "error" in kwargs:
            self.error = kwargs["error"]

        return old_status

    @property
    def duration_ms(self) -> int | None:
        if not self.started_at or not self.completed_at:
            return None
        delta = self.completed_at - self.started_at
        return int(delta.total_seconds() * 1000)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_pending(self) -> bool:
        return self.status == Status.PENDING

    @property
    def is_executing(self) -> bool:
        return self.status == Status.EXECUTING

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "agent": self.agent,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class TaskManager:
    def __init__(self, hook_registry: HookRegistry | None = None) -> None:
        self._tasks: dict[str, Task] = {}
        self._hooks = hook_registry

    def _emit(self, event: Any) -> None:
        if not self._hooks or not self._hooks.has_callbacks():
            return
        self._hooks.invoke_callbacks(event)

    def create(
        self,
        name: str,
        agent: str,
        description: str | None = None,
        depends_on: list[str] | None = None,
    ) -> Task:
        if name in self._tasks:
            raise ValueError(f"Task '{name}' already exists")

        depends_on_list = depends_on or []
        task = Task(
            name=name,
            agent=agent,
            description=description,
            depends_on=depends_on_list,
        )
        self._tasks[name] = task

        self._emit(
            TaskCreatedEvent(
                name=name,
                agent=agent,
                description=description,
                depends_on=depends_on_list,
            )
        )

        return task

    def start(self, name: str) -> None:
        task = self._tasks[name]
        task.transition_to(Status.EXECUTING)

        self._emit(TaskStartedEvent(name=name))

    def complete(self, name: str, result: Any = None) -> None:
        task = self._tasks[name]
        task.transition_to(Status.COMPLETED, result=result)

        self._emit(TaskCompletedEvent(name=name, result=result))

    def fail(self, name: str, error: str) -> None:
        task = self._tasks[name]
        task.transition_to(Status.FAILED, error=error)

        self._emit(TaskFailedEvent(name=name, error=error))

    def interrupt(self, name: str, reason: str | None = None) -> None:
        task = self._tasks[name]
        task.transition_to(Status.INTERRUPTED, error=reason)

        self._emit(TaskInterruptedEvent(name=name, reason=reason))

    def get(self, name: str) -> Task | None:
        return self._tasks.get(name)

    def by_status(self, status: Status) -> list[Task]:
        return [t for t in self._tasks.values() if t.status == status]

    @property
    def all_tasks(self) -> dict[str, Task]:
        return dict(self._tasks)

    @property
    def all_terminal(self) -> bool:
        return all(t.is_terminal for t in self._tasks.values())

    @property
    def pending_count(self) -> int:
        return len(self.by_status(Status.PENDING))

    @property
    def executing_count(self) -> int:
        return len(self.by_status(Status.EXECUTING))

    @property
    def completed_count(self) -> int:
        return len(self.by_status(Status.COMPLETED))

    @property
    def failed_count(self) -> int:
        return len(self.by_status(Status.FAILED))

    def clear(self) -> None:
        self._tasks.clear()

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, name: str) -> bool:
        return name in self._tasks


__all__ = [
    "Task",
    "TaskManager",
]
