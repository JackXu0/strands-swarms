"""Event types for DynamicSwarm execution.

This module extends strands' hook system with events specific to dynamic
swarm planning and execution. Events follow the same patterns as
strands.hooks.events for consistency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from strands.hooks.registry import BaseHookEvent, HookProvider, HookRegistry
from strands.multiagent.base import Status

from .definition import AGENT_COLORS

if TYPE_CHECKING:
    from .definition import AgentDefinition, TaskDefinition

# =============================================================================
# Planning Events
# =============================================================================


@dataclass
class SwarmStartedEvent(BaseHookEvent):
    """Event triggered when a dynamic swarm begins execution.
    
    Attributes:
        query: The user's input query.
        available_tools: List of tool names available to spawned agents.
        available_models: List of model names available to spawned agents.
    """
    
    query: str
    available_tools: list[str] = field(default_factory=list)
    available_models: list[str] = field(default_factory=list)


@dataclass
class PlanningStartedEvent(BaseHookEvent):
    """Event triggered when the planning phase begins."""
    
    pass


@dataclass
class AgentSpawnedEvent(BaseHookEvent):
    """Event triggered when a new agent is dynamically spawned.
    
    Attributes:
        name: The agent's unique identifier.
        role: The agent's role description.
        instructions: Additional instructions for the agent.
        tools: List of tool names assigned to this agent.
        model: Model name assigned to this agent.
        color: ANSI color code assigned to this agent.
    """
    
    name: str
    role: str
    instructions: str | None = None
    tools: list[str] = field(default_factory=list)
    model: str | None = None
    color: str | None = None  # ANSI color code for consistent display


@dataclass
class TaskCreatedEvent(BaseHookEvent):
    """Event triggered when a new task is created.
    
    Attributes:
        name: The task's unique identifier.
        agent: Name of the agent assigned to this task.
        description: Description of what the task does.
        depends_on: List of task names this task depends on.
    """
    
    name: str
    agent: str
    description: str | None = None
    depends_on: list[str] = field(default_factory=list)


@dataclass
class PlanningCompletedEvent(BaseHookEvent):
    """Event triggered when planning phase completes.

    Attributes:
        entry_task: The entry point task, if specified.
        agents: List of AgentDefinition for summary display.
        tasks: List of TaskDefinition for summary display.
    """

    entry_task: str | None = None
    agents: list["AgentDefinition"] = field(default_factory=list)
    tasks: list["TaskDefinition"] = field(default_factory=list)


# =============================================================================
# Execution Events
# =============================================================================


@dataclass
class ExecutionStartedEvent(BaseHookEvent):
    """Event triggered when the execution phase begins.
    
    Attributes:
        tasks: List of task names to execute.
    """
    
    tasks: list[str] = field(default_factory=list)


@dataclass
class TaskStartedEvent(BaseHookEvent):
    """Event triggered when a task begins executing.
    
    Attributes:
        name: The task name.
    """
    
    name: str


@dataclass
class TaskCompletedEvent(BaseHookEvent):
    """Event triggered when a task completes successfully.
    
    Attributes:
        name: The task name.
        result: Optional result from task execution.
    """
    
    name: str
    result: Any = None

    @property
    def should_reverse_callbacks(self) -> bool:
        """Cleanup events should reverse callback order."""
        return True


@dataclass
class TaskFailedEvent(BaseHookEvent):
    """Event triggered when a task fails.
    
    Attributes:
        name: The task name.
        error: Error message or description.
    """
    
    name: str
    error: str | None = None

    @property
    def should_reverse_callbacks(self) -> bool:
        """Cleanup events should reverse callback order."""
        return True


@dataclass
class TaskInterruptedEvent(BaseHookEvent):
    """Event triggered when a task is interrupted.
    
    Attributes:
        name: The task name.
        reason: Optional reason for interruption.
    """
    
    name: str
    reason: str | None = None

    @property
    def should_reverse_callbacks(self) -> bool:
        """Cleanup events should reverse callback order."""
        return True


@dataclass
class ExecutionCompletedEvent(BaseHookEvent):
    """Event triggered when execution phase completes.
    
    Attributes:
        status: Final execution status.
        agent_count: Number of agents used.
        task_count: Number of tasks completed.
    """
    
    status: str
    agent_count: int = 0
    task_count: int = 0

    @property
    def should_reverse_callbacks(self) -> bool:
        """Cleanup events should reverse callback order."""
        return True


@dataclass
class SwarmCompletedEvent(BaseHookEvent):
    """Event triggered when swarm completes successfully."""
    
    pass

    @property
    def should_reverse_callbacks(self) -> bool:
        """Cleanup events should reverse callback order."""
        return True


@dataclass
class SwarmFailedEvent(BaseHookEvent):
    """Event triggered when swarm fails.
    
    Attributes:
        error: Error message or description.
    """
    
    error: str

    @property
    def should_reverse_callbacks(self) -> bool:
        """Cleanup events should reverse callback order."""
        return True


# =============================================================================
# Default Hook Provider
# =============================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


# =============================================================================
# Colored Callback Handler
# =============================================================================


# Patterns for internal LLM reasoning tags that should be filtered from output
_INTERNAL_TAG_PATTERN = re.compile(
    r'<(/?)(?:thinking|result|task_quality_reflection|task_quality_score|'
    r'search_quality_reflection|search_quality_score|reflection|score)>',
    re.IGNORECASE
)


def _filter_internal_tags(text: str) -> str:
    """Filter out internal LLM reasoning tags from streamed text.
    
    These tags are generated by the LLM for internal reasoning but shouldn't
    be displayed to users in verbose output.
    """
    return _INTERNAL_TAG_PATTERN.sub('', text)


def create_colored_callback_handler(color: str, agent_name: str) -> Callable[..., None]:
    """Create a callback handler that prints agent output with a specific color.
    
    This ensures all output from a specific agent (text, reasoning, tool calls)
    is displayed with the same color for easy visual tracking.
    
    Internal reasoning tags (like <thinking>, <result>, etc.) are automatically
    filtered out from the streamed output.
    
    Args:
        color: ANSI color code to use for this agent's output.
        agent_name: Name of the agent (for tool call headers).
    
    Returns:
        A callback handler function for the strands Agent.
    """
    # Track which tool calls we've already printed to avoid duplicates
    # (strands fires current_tool_use multiple times as input streams in)
    printed_tool_ids: set[str] = set()
    
    def handler(**kwargs: Any) -> None:
        nonlocal printed_tool_ids
        
        # Text generation events
        if "data" in kwargs:
            # Filter internal reasoning tags and stream text chunk with color
            filtered_text = _filter_internal_tags(kwargs['data'])
            if filtered_text:
                print(f"{color}{filtered_text}{RESET}", end="", flush=True)
        
        # Tool usage events - only print once per tool call
        elif "current_tool_use" in kwargs:
            tool = kwargs["current_tool_use"]
            tool_id = tool.get("toolUseId", "")
            
            # Skip if we've already printed this tool call
            if tool_id in printed_tool_ids:
                return
            printed_tool_ids.add(tool_id)
            
            tool_name = tool.get("name", "unknown")
            # Print tool header with color
            print(f"\n{color}{BOLD}Tool: {tool_name}{RESET}")
        
        # Tool result events - print the input when tool completes
        elif "tool_result" in kwargs:
            result = kwargs["tool_result"]
            tool_id = result.get("toolUseId", "")
            tool_input = result.get("input")
            
            if tool_input:
                if isinstance(tool_input, dict):
                    for key, value in tool_input.items():
                        # Truncate long values for readability
                        str_value = str(value)
                        if len(str_value) > 100:
                            str_value = str_value[:100] + "..."
                        print(f"{color}  {key}: {str_value}{RESET}")
                else:
                    str_input = str(tool_input)
                    if len(str_input) > 200:
                        str_input = str_input[:200] + "..."
                    print(f"{color}  Input: {str_input}{RESET}")
        
        # Reasoning/thinking events - also filter internal tags
        elif "reasoningText" in kwargs:
            filtered_text = _filter_internal_tags(kwargs['reasoningText'])
            if filtered_text:
                print(f"{color}{filtered_text}{RESET}", end="", flush=True)
        elif "reasoning" in kwargs:
            # Start of reasoning block
            pass  # The reasoningText will contain the actual content
    
    return handler


class PrintingHookProvider(HookProvider):
    """Hook provider that prints formatted output to console.
    
    This provides a nice CLI experience with emoji indicators and colors.
    Use with DynamicSwarm for live status updates.
    
    Example:
        swarm = DynamicSwarm(
            ...,
            hooks=[PrintingHookProvider()],
        )
    """
    
    # Use same colors as agent output for consistency
    COLORS = AGENT_COLORS
    
    def __init__(self, use_colors: bool = True) -> None:
        """Initialize the hook provider with state tracking.
        
        Args:
            use_colors: Whether to use ANSI colors for agent differentiation.
        """
        self._agents_header_printed = False
        self._tasks_header_printed = False
        self._use_colors = use_colors
        self._agent_colors: dict[str, str] = {}  # Populated from AgentSpawnedEvent
        self._task_agents: dict[str, str] = {}   # task_name -> agent_name
    
    def _get_agent_color(self, agent_name: str | None) -> str:
        """Get a consistent color for an agent."""
        if not self._use_colors or not agent_name:
            return ""
        return self._agent_colors.get(agent_name, "")
    
    def _colored(self, text: str, agent_name: str | None = None, bold: bool = False) -> str:
        """Apply color to text."""
        if not self._use_colors:
            return text
        color = self._get_agent_color(agent_name) if agent_name else ""
        bold_code = BOLD if bold else ""
        return f"{bold_code}{color}{text}{RESET}"
    
    def _get_task_agent(self, task_name: str) -> str | None:
        """Get the agent assigned to a task."""
        return self._task_agents.get(task_name)
    
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register all printing callbacks."""
        # Planning events
        registry.add_callback(SwarmStartedEvent, self._on_swarm_started)
        registry.add_callback(PlanningStartedEvent, self._on_planning_started)
        registry.add_callback(AgentSpawnedEvent, self._on_agent_spawned)
        registry.add_callback(TaskCreatedEvent, self._on_task_created)
        registry.add_callback(PlanningCompletedEvent, self._on_planning_completed)
        
        # Execution events
        registry.add_callback(ExecutionStartedEvent, self._on_execution_started)
        registry.add_callback(TaskStartedEvent, self._on_task_started)
        registry.add_callback(TaskCompletedEvent, self._on_task_completed)
        registry.add_callback(TaskFailedEvent, self._on_task_failed)
        registry.add_callback(TaskInterruptedEvent, self._on_task_interrupted)
        registry.add_callback(ExecutionCompletedEvent, self._on_execution_completed)
        
        # Completion events
        registry.add_callback(SwarmCompletedEvent, self._on_swarm_completed)
        registry.add_callback(SwarmFailedEvent, self._on_swarm_failed)
    
    def _on_swarm_started(self, event: SwarmStartedEvent) -> None:
        # Reset state for new swarm execution
        self._agents_header_printed = False
        self._tasks_header_printed = False
        self._agent_colors.clear()
        self._task_agents.clear()
        
        print("\n" + "=" * 60)
        print("🚀 DYNAMIC SWARM STARTING")
        print("=" * 60)
        query = event.query
        print(f"\n📝 Query: {query[:200]}{'...' if len(query) > 200 else ''}")
        print(f"📦 Available tools: {event.available_tools or ['none']}")
        print(f"🧠 Available models: {event.available_models or ['default']}")
    
    def _on_planning_started(self, event: PlanningStartedEvent) -> None:
        print("\n" + "-" * 40)
        print("📐 PHASE 1: PLANNING")
        print("-" * 40)
    
    def _on_agent_spawned(self, event: AgentSpawnedEvent) -> None:
        # Store the color from the event (assigned by registry) for later use
        # Don't print here - we'll print a clean summary after planning completes
        if event.color:
            self._agent_colors[event.name] = event.color
    
    def _on_task_created(self, event: TaskCreatedEvent) -> None:
        # Track task-agent mapping for coloring during execution
        # Don't print here - we'll print a clean summary after planning completes
        self._task_agents[event.name] = event.agent
    
    def _on_planning_completed(self, event: PlanningCompletedEvent) -> None:
        # Print agents summary
        if event.agents:
            print("\n" + "·" * 40)
            print("🤖 AGENTS")
            print("·" * 40)
            for agent in event.agents:
                # Store color for later use
                if agent.color:
                    self._agent_colors[agent.name] = agent.color
                
                # Color the entire agent block
                c = self._colored
                print(f"\n  {c(f'[{agent.name}]', agent.name, bold=True)}")
                print(f"    {c('Role:', agent.name, bold=True)} {c(agent.role, agent.name)}")
                print(f"    {c('Tools:', agent.name, bold=True)} {c(str(agent.tools or ['none']), agent.name)}")
                print(f"    {c('Model:', agent.name, bold=True)} {c(agent.model or 'default', agent.name)}")
        
        # Print tasks summary with dependency visualization
        if event.tasks:
            print("\n" + "·" * 40)
            print("📋 TASKS & DEPENDENCIES")
            print("·" * 40)
            for task in event.tasks:
                # Track task-agent mapping
                self._task_agents[task.name] = task.agent
                
                # Color the entire task block with the agent's color
                c = self._colored
                agent_name = task.agent
                
                print(f"\n  {c(f'[{task.name}]', agent_name, bold=True)}")
                print(f"    {c('Agent:', agent_name, bold=True)} {c(agent_name, agent_name)}")
                if task.description:
                    print(f"    {c('Description:', agent_name, bold=True)} {c(task.description, agent_name)}")
                if task.depends_on:
                    deps_colored = [self._colored(d, self._task_agents.get(d), bold=True) for d in task.depends_on]
                    print(f"    {c('⏳ Waits for:', agent_name, bold=True)} {', '.join(deps_colored)}")
                else:
                    print(f"    {c('⚡ Can start immediately', agent_name)}")
        
        # Print execution summary
        print("\n" + "·" * 40)
        print("✅ PLAN READY")
        print("·" * 40)
        print(f"  Entry: {event.entry_task or 'auto'}")
        print(f"  Total: {len(event.agents)} agents, {len(event.tasks)} tasks")
    
    def _on_execution_started(self, event: ExecutionStartedEvent) -> None:
        print("\n" + "-" * 40)
        print("⚡ PHASE 2: EXECUTION")
        print("-" * 40)
        # Color each task name by its agent
        tasks_colored = [self._colored(t, self._get_task_agent(t), bold=True) for t in event.tasks]
        print(f"📋 Tasks to execute: [{', '.join(tasks_colored)}]")
    
    def _on_task_started(self, event: TaskStartedEvent) -> None:
        agent = self._get_task_agent(event.name)
        task_str = self._colored(event.name, agent, bold=True)
        print(f"\n▶️  Executing task: {task_str}")
    
    def _on_task_completed(self, event: TaskCompletedEvent) -> None:
        agent = self._get_task_agent(event.name)
        task_str = self._colored(event.name, agent)
        print(f"   ✓ Completed: {task_str}")
    
    def _on_task_failed(self, event: TaskFailedEvent) -> None:
        agent = self._get_task_agent(event.name)
        task_str = self._colored(event.name, agent)
        print(f"   ❌ Failed: {task_str}")
        if event.error:
            print(f"   Error: {event.error}")
    
    def _on_task_interrupted(self, event: TaskInterruptedEvent) -> None:
        agent = self._get_task_agent(event.name)
        task_str = self._colored(event.name, agent)
        print(f"   ⏸️  Interrupted: {task_str}")
    
    def _on_execution_completed(self, event: ExecutionCompletedEvent) -> None:
        print("\n" + "-" * 40)
        print("🏁 EXECUTION COMPLETE")
        print("-" * 40)
        print(f"   Status: {event.status}")
        print(f"   Agents used: {event.agent_count}")
        print(f"   Tasks completed: {event.task_count}")
    
    def _on_swarm_completed(self, event: SwarmCompletedEvent) -> None:
        print("\n" + "=" * 60)
        print("✅ SWARM COMPLETED SUCCESSFULLY")
        print("=" * 60)
    
    def _on_swarm_failed(self, event: SwarmFailedEvent) -> None:
        print("\n" + "=" * 60)
        print(f"❌ SWARM FAILED: {event.error}")
        print("=" * 60)


# Re-export strands types for convenience
__all__ = [
    # Events
    "SwarmStartedEvent",
    "PlanningStartedEvent",
    "AgentSpawnedEvent",
    "TaskCreatedEvent",
    "PlanningCompletedEvent",
    "ExecutionStartedEvent",
    "TaskStartedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "TaskInterruptedEvent",
    "ExecutionCompletedEvent",
    "SwarmCompletedEvent",
    "SwarmFailedEvent",
    # Hook provider
    "PrintingHookProvider",
    # Callback handler factory
    "create_colored_callback_handler",
    # Color constants
    "AGENT_COLORS",
    "RESET",
    "BOLD",
    "DIM",
    # Re-exports from strands
    "HookProvider",
    "HookRegistry",
    "Status",
]
