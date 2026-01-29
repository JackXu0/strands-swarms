"""Runtime registry for dynamically spawned agents and tasks."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from strands import Agent

if TYPE_CHECKING:
    from strands.models import Model


# =============================================================================
# Colored Callback Handler for Agent Output
# =============================================================================

# ANSI color codes
AGENT_COLORS = [
    "\033[94m",   # Blue
    "\033[92m",   # Green
    "\033[93m",   # Yellow
    "\033[95m",   # Magenta
    "\033[96m",   # Cyan
    "\033[91m",   # Red
]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def create_colored_callback(agent_name: str, color: str) -> Callable[..., None]:
    """Create a callback handler that colors ALL output for a specific agent.
    
    This makes it easy to track which agent is producing which output during
    async execution, since everything from one agent will be in the same color.
    
    Args:
        agent_name: Name of the agent (for prefix).
        color: ANSI color code to use.
    
    Returns:
        Callback function that prints colored output.
    """
    # Track state for formatting
    state = {"started": False, "has_output": False, "in_tool": False}
    
    def colored_callback(**kwargs: Any) -> None:
        # Print agent prefix at start of new output block
        def ensure_started():
            if not state["started"]:
                sys.stdout.write(f"\n{color}{BOLD}[{agent_name}]{RESET} ")
                state["started"] = True
        
        # Handle streaming text data
        if "data" in kwargs:
            data = kwargs["data"]
            ensure_started()
            sys.stdout.write(f"{color}{data}{RESET}")
            sys.stdout.flush()
            state["has_output"] = True
        
        # Handle tool use - show which tool is being called
        if "current_tool_use" in kwargs and not state["in_tool"]:
            tool_use = kwargs["current_tool_use"]
            tool_name = tool_use.get("name", "unknown") if isinstance(tool_use, dict) else getattr(tool_use, "name", "unknown")
            ensure_started()
            sys.stdout.write(f"{DIM}{color}[calling {tool_name}]{RESET} ")
            sys.stdout.flush()
            state["in_tool"] = True
            state["has_output"] = True
        
        # Handle tool result
        if "tool_result" in kwargs:
            state["in_tool"] = False
        
        # Handle completion - reset state for next call
        if kwargs.get("complete", False):
            if state["has_output"]:
                sys.stdout.write("\n")
                sys.stdout.flush()
            state["started"] = False
            state["has_output"] = False
            state["in_tool"] = False
    
    return colored_callback


@dataclass
class AgentDefinition:
    """Definition for a dynamically spawned agent.

    Attributes:
        name: Unique identifier for this agent.
        role: What this agent does (used in system prompt).
        instructions: Additional instructions for the agent.
        tools: List of tool names from the available pool.
        model: Model name from the available pool.
    """

    name: str
    role: str
    instructions: str | None = None
    tools: list[str] = field(default_factory=list)
    model: str | None = None

    def build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        parts = [f"You are a {self.role}."]
        if self.instructions:
            parts.append(f"\n\nInstructions:\n{self.instructions}")
        return "\n".join(parts)


@dataclass
class TaskDefinition:
    """Definition for a task in the dynamic workflow.

    Attributes:
        name: Unique identifier for this task.
        agent: Name of the agent assigned to this task.
        description: What this task should accomplish.
        depends_on: List of task names this task depends on.
    """

    name: str
    agent: str
    description: str | None = None
    depends_on: list[str] = field(default_factory=list)


class DynamicRegistry:
    """Registry that collects agent and task definitions during planning.

    The planner agent uses tools that populate this registry. After planning,
    the registry is used to build the actual swarm/graph.
    """

    def __init__(
        self,
        available_tools: dict[str, Callable[..., Any]],
        available_models: dict[str, Model],
        default_model: str | None = None,
        use_colors: bool = True,
    ) -> None:
        """Initialize the registry.

        Args:
            available_tools: Pool of tools that can be assigned to agents.
            available_models: Pool of models (name -> Model instance) that can be used by agents.
            default_model: Default model name to use if none specified.
            use_colors: Whether to use colored output for agent streaming.
        """
        self._available_tools = available_tools
        self._available_models = available_models
        self._default_model = default_model or next(iter(available_models.keys()), None)
        self._use_colors = use_colors

        self._agents: dict[str, AgentDefinition] = {}
        self._tasks: dict[str, TaskDefinition] = {}
        self._entry_task: str | None = None
        self._execution_mode: str = "swarm"
        
        # Color assignment for agents
        self._agent_colors: dict[str, str] = {}
        self._color_index = 0

    def register_agent(self, definition: AgentDefinition) -> None:
        """Register a new agent definition.

        Args:
            definition: The agent definition to register.

        Raises:
            ValueError: If agent name already exists or references invalid tools/models.
        """
        if definition.name in self._agents:
            raise ValueError(f"Agent '{definition.name}' already exists")

        # Validate tools
        for tool_name in definition.tools:
            if tool_name not in self._available_tools:
                available = list(self._available_tools.keys())
                raise ValueError(
                    f"Tool '{tool_name}' not in available tools: {available}"
                )

        # Validate model
        if definition.model and definition.model not in self._available_models:
            available = list(self._available_models.keys())
            raise ValueError(
                f"Model '{definition.model}' not in available models: {available}"
            )

        self._agents[definition.name] = definition
        
        # Assign a color to this agent
        if self._use_colors:
            self._agent_colors[definition.name] = AGENT_COLORS[self._color_index % len(AGENT_COLORS)]
            self._color_index += 1

    def register_task(self, definition: TaskDefinition) -> None:
        """Register a new task definition.

        Args:
            definition: The task definition to register.

        Raises:
            ValueError: If task name exists or references unknown agent/dependencies.
        """
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

    def set_entry_task(self, task_name: str) -> None:
        """Set the entry point task."""
        if task_name not in self._tasks:
            raise ValueError(f"Task '{task_name}' not found")
        self._entry_task = task_name

    def set_execution_mode(self, mode: str) -> None:
        """Set execution mode (swarm or graph)."""
        if mode not in ("swarm", "graph"):
            raise ValueError(f"Mode must be 'swarm' or 'graph', got '{mode}'")
        self._execution_mode = mode

    def build_agent(self, agent_name: str) -> Agent:
        """Build an actual Agent from a registered definition.

        Args:
            agent_name: Name of the registered agent.

        Returns:
            Configured Agent instance.
        """
        definition = self._agents.get(agent_name)
        if not definition:
            raise ValueError(f"Agent '{agent_name}' not found")

        # Resolve tools
        tools = [self._available_tools[t] for t in definition.tools]

        # Resolve model - get the Model instance from the pool
        model_name = definition.model or self._default_model
        model = self._available_models.get(model_name) if model_name else None
        
        # Create colored callback handler if colors are enabled
        callback_handler = None
        if self._use_colors and agent_name in self._agent_colors:
            callback_handler = create_colored_callback(
                agent_name, 
                self._agent_colors[agent_name]
            )

        return Agent(
            name=definition.name,
            system_prompt=definition.build_system_prompt(),
            model=model,
            tools=tools if tools else None,
            callback_handler=callback_handler,
        )

    def clear(self) -> None:
        """Clear all registered agents and tasks."""
        self._agents.clear()
        self._tasks.clear()
        self._agent_colors.clear()
        self._color_index = 0
        self._entry_task = None
        self._execution_mode = "swarm"

    @property
    def agents(self) -> dict[str, AgentDefinition]:
        """Get all registered agent definitions."""
        return dict(self._agents)

    @property
    def tasks(self) -> dict[str, TaskDefinition]:
        """Get all registered task definitions."""
        return dict(self._tasks)

    @property
    def entry_task(self) -> str | None:
        """Get the entry task name."""
        return self._entry_task

    @property
    def execution_mode(self) -> str:
        """Get the execution mode."""
        return self._execution_mode

    @property
    def available_tool_names(self) -> list[str]:
        """Get list of available tool names."""
        return list(self._available_tools.keys())

    @property
    def available_model_names(self) -> list[str]:
        """Get list of available model names."""
        return list(self._available_models.keys())
    
    @property
    def agent_colors(self) -> dict[str, str]:
        """Get agent name to color mapping."""
        return dict(self._agent_colors)
    
    def get_agent_color(self, agent_name: str) -> str | None:
        """Get the color assigned to an agent."""
        return self._agent_colors.get(agent_name)

    def get_summary(self) -> str:
        """Get a summary of the current registry state."""
        lines = [
            f"Agents ({len(self._agents)}):",
            *[f"  - {name}: {d.role}" for name, d in self._agents.items()],
            f"\nTasks ({len(self._tasks)}):",
            *[
                f"  - {name} -> {d.agent}" + (f" (depends: {d.depends_on})" if d.depends_on else "")
                for name, d in self._tasks.items()
            ],
            f"\nMode: {self._execution_mode}",
            f"Entry: {self._entry_task or 'auto'}",
        ]
        return "\n".join(lines)
