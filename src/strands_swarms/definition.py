"""Swarm definition types - data structures for defining multi-agent workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from strands.hooks import HookRegistry
from strands.session.file_session_manager import FileSessionManager

from .events import AGENT_COLORS

if TYPE_CHECKING:
    from strands.models import Model


@dataclass(frozen=True)
class DynamicSwarmCapabilities:
    """Immutable configuration for available tools and models."""

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


@dataclass
class AgentDefinition:
    """Definition of a sub-agent to be spawned."""

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
    """Definition of a task to be executed by a sub-agent."""

    name: str
    agent: str
    description: str | None = None
    depends_on: list[str] = field(default_factory=list)


@dataclass
class SessionConfig:
    """Configuration for session persistence."""

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


class SwarmDefinition:
    """Per-query definition of sub-agents and tasks created during planning."""

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
