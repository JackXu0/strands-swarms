"""Planning tools and planner agent for dynamic swarm construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strands import Agent, tool
from strands.hooks import HookRegistry

from .events import AgentSpawnedEvent, TaskCreatedEvent, PlanningCompletedEvent, AgentInfo, TaskInfo

if TYPE_CHECKING:
    from strands.models import Model
    from .registry import DynamicRegistry


# Module-level registry reference (set by DynamicSwarm before planning)
_active_registry: DynamicRegistry | None = None
_hook_registry: HookRegistry | None = None


def set_active_registry(
    registry: DynamicRegistry | None,
    hook_registry: HookRegistry | None = None,
) -> None:
    """Set the active registry and hook registry for planning tools."""
    global _active_registry, _hook_registry
    _active_registry = registry
    _hook_registry = hook_registry


def get_active_registry() -> DynamicRegistry:
    """Get the active registry, raising if not set."""
    if _active_registry is None:
        raise RuntimeError("No active registry. Planning tools must be used within DynamicSwarm.")
    return _active_registry


def _emit(event: Any) -> None:
    """Emit an event to all registered hooks."""
    if _hook_registry and _hook_registry.has_callbacks():
        _hook_registry.invoke_callbacks(event)


# =============================================================================
# Planning Tools
# =============================================================================


@tool
def spawn_agent(
    name: str,
    role: str,
    instructions: str = "",
    tools: list[str] | None = None,
    model: str | None = None,
) -> str:
    """Spawn a new sub-agent with specific capabilities.

    Use this to create specialized agents for different parts of the workflow.
    Each agent can have its own tools and model optimized for its role.

    Args:
        name: Unique identifier for this agent (e.g., "researcher", "coder").
        role: What this agent does - becomes its core identity (e.g., "Research and gather information from the web").
        instructions: Additional detailed instructions for how the agent should behave.
        tools: List of tool names from the available pool. Use list_available_tools() to see options.
        model: Model identifier from the available pool. Use list_available_models() to see options.
               If not specified, uses the default model.

    Returns:
        Confirmation message with agent details.

    Example:
        spawn_agent(
            name="researcher",
            role="Research topics thoroughly using web search",
            tools=["search_web", "read_url"],
            model="powerful"
        )
    """
    from .registry import AgentDefinition

    registry = get_active_registry()

    definition = AgentDefinition(
        name=name,
        role=role,
        instructions=instructions if instructions else None,
        tools=tools or [],
        model=model,
    )

    try:
        registry.register_agent(definition)
        tool_info = f"tools={tools}" if tools else "no tools"
        model_info = f"model={model}" if model else "default model"
        
        # Emit event with color from registry for consistent display
        _emit(AgentSpawnedEvent(
            name=name,
            role=role,
            instructions=instructions or None,
            tools=tools or [],
            model=model,
            color=registry.get_agent_color(name),
        ))
        
        return f"✓ Spawned agent '{name}' ({role}) with {tool_info}, {model_info}"
    except ValueError as e:
        return f"✗ Failed to spawn agent: {e}"


@tool
def create_task(
    name: str,
    agent_name: str,
    description: str = "",
    depends_on: list[str] | None = None,
) -> str:
    """Create a task and assign it to a spawned agent.

    Args:
        name: Unique identifier for this task (e.g., "research_task").
        agent_name: Name of the agent to assign this task to. Must exactly match an agent name from spawn_agent().
        description: What this task should accomplish.
        depends_on: List of task names that must complete before this task.

    Returns:
        Confirmation message with task details.

    Example:
        create_task(name="research_task", agent_name="researcher", description="Research AI trends")
    """
    from .registry import TaskDefinition

    registry = get_active_registry()

    definition = TaskDefinition(
        name=name,
        agent=agent_name,
        description=description if description else None,
        depends_on=depends_on or [],
    )

    try:
        registry.register_task(definition)
        deps_info = f"depends_on={depends_on}" if depends_on else "no dependencies"
        
        # Emit event
        _emit(TaskCreatedEvent(
            name=name,
            agent=agent_name,
            description=description or None,
            depends_on=depends_on or [],
        ))
        
        return f"✓ Created task '{name}' -> agent '{agent_name}' ({deps_info})"
    except ValueError as e:
        return f"✗ Failed to create task: {e}"


@tool
def execute_swarm(
    mode: str = "swarm",
    entry_task: str | None = None,
) -> str:
    """Signal that planning is complete and the swarm should be executed.

    Call this after you have spawned all necessary agents and created all tasks.
    The system will build and execute the workflow based on your definitions.

    Args:
        mode: Execution mode - "swarm" for dynamic handoffs, "graph" for
              deterministic dependency-based execution. Use "graph" when
              you have clear dependencies. Use "swarm" when agents need
              flexibility to hand off dynamically.
        entry_task: Optional name of the task to start with. If not specified,
                   the system will determine entry points automatically.

    Returns:
        Confirmation that execution will proceed.

    Example:
        execute_swarm(mode="graph", entry_task="research")
    """
    registry = get_active_registry()

    try:
        registry.set_execution_mode(mode)
        if entry_task:
            registry.set_entry_task(entry_task)

        summary = registry.get_summary()
        
        # Build agent info list for summary display
        agents_info = [
            AgentInfo(
                name=agent_def.name,
                role=agent_def.role,
                tools=agent_def.tools,
                model=agent_def.model,
                color=registry.get_agent_color(agent_def.name),
            )
            for agent_def in registry.agents.values()
        ]
        
        # Build task info list for summary display
        tasks_info = [
            TaskInfo(
                name=task_def.name,
                agent=task_def.agent,
                description=task_def.description,
                depends_on=task_def.depends_on,
            )
            for task_def in registry.tasks.values()
        ]
        
        # Emit event with full planning info
        _emit(PlanningCompletedEvent(
            mode=mode,
            entry_task=entry_task,
            agents=agents_info,
            tasks=tasks_info,
        ))
        
        return f"✓ Planning complete. Ready to execute.\n\n{summary}"
    except ValueError as e:
        return f"✗ Failed to configure execution: {e}"


@tool
def list_available_tools() -> str:
    """List all tools available for spawned agents.

    Returns:
        List of available tool names that can be assigned to agents.
    """
    registry = get_active_registry()
    tools = registry.available_tool_names
    if not tools:
        return "No tools available."
    return "Available tools:\n" + "\n".join(f"  - {t}" for t in tools)


@tool
def list_available_models() -> str:
    """List all models available for spawned agents.

    Returns:
        List of available model identifiers.
    """
    registry = get_active_registry()
    models = registry.available_model_names
    if not models:
        return "No models configured. Default model will be used."
    return "Available models:\n" + "\n".join(f"  - {m}" for m in models)


@tool
def get_planning_status() -> str:
    """Get current status of the planning - agents and tasks defined so far.

    Returns:
        Summary of current agents and tasks.
    """
    registry = get_active_registry()
    return registry.get_summary()


# =============================================================================
# Planner Agent
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are a workflow planner that designs multi-agent workflows.

## Your Process

Think through the problem step-by-step:

1. **Analyze the request** - What needs to be done? Break it into logical steps.
2. **Design agents** - What specialized agents are needed? What tools does each need?
3. **Plan dependencies** - Which tasks must wait for others? Draw the dependency graph.
4. **Execute** - Run the workflow.

## Output Format

Structure your response clearly:

```
### Analysis
[Brief analysis of what's needed]

### Agents Needed
[List agents you'll create and why]

### Workflow Dependencies  
[Show which tasks depend on which, e.g., "write_report waits for research to complete"]

### Building Workflow
[Then call the tools]
```

## Tools

**spawn_agent(name, role, tools, model)**
- Creates an agent with specific capabilities

**create_task(name, agent_name, description, depends_on)**  
- Creates a task assigned to an agent
- depends_on: list of task names that must finish BEFORE this task starts

**execute_swarm(mode, entry_task)**
- mode="graph" for dependency-based execution (task B waits for task A)
- mode="swarm" for dynamic handoffs (agents decide who goes next)

## Example

For "Research AI and write a report":

### Analysis
Need to: 1) Research AI trends, 2) Write a report based on research.
The report cannot be written until research is done.

### Agents Needed
- researcher: Gathers information (needs search_web)
- writer: Creates reports (needs write_file)

### Workflow Dependencies
research_task → write_task (writer waits for researcher)

### Building Workflow
[spawn agents, create tasks with depends_on, execute]

Now analyze the request and build the workflow."""


def create_planner_agent(model: Model | None = None) -> Agent:
    """Create the planner agent with planning tools.

    Args:
        model: Model instance to use for the planner. Defaults to None (use default).

    Returns:
        Configured planner Agent.
    """
    planning_tools = [
        spawn_agent,
        create_task,
        execute_swarm,
        list_available_tools,
        list_available_models,
        get_planning_status,
    ]
    
    return Agent(
        name="planner",
        system_prompt=PLANNER_SYSTEM_PROMPT,
        model=model,
        tools=planning_tools,
    )
