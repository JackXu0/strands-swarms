"""Orchestrator agent that plans, creates sub-agents, assigns tasks, and synthesizes completion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from strands import Agent, tool

from .events import AgentSpawnedEvent, PlanningCompletedEvent, TaskCreatedEvent

if TYPE_CHECKING:
    from strands.models import Model

    from .swarm import SwarmDefinition


# --- System Prompt ---

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a workflow orchestrator that plans, creates sub-agents, assigns tasks, and synthesizes completion.

## Your Four Responsibilities

### 1. Planning
Analyze the request and break it into logical steps. Determine what specialized agents are needed.

### 2. Creating Sub-agents
Spawn specialized agents with appropriate tools and models. Each agent should have a focused role.

### 3. Assigning Tasks
Create tasks and assign them to agents. Define dependencies when one task needs results from another.

### 4. Synthesizing Completion
After all tasks complete, synthesize outputs into a cohesive final response. Be direct - deliver
the result as if you completed the task yourself without mentioning sub-agents or the process.

## Your Process

Think through the problem step-by-step:

1. **Analyze the request** - What needs to be done? Break it into logical steps.
2. **Design agents** - What specialized agents are needed? What tools does each need?
3. **Plan dependencies** - Which tasks must wait for others? Draw the dependency graph.
4. **Finalize** - Call finalize_plan when done creating agents and tasks.

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

**finalize_plan()**
- Signals that planning is complete
- Call this after you've created all agents and tasks

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
[spawn agents, create tasks with depends_on, finalize_plan]

Now analyze the request and build the workflow.
"""


def create_orchestrator_tools(context: "SwarmDefinition") -> list[Callable[..., str]]:
    """Create orchestrator tools that capture the SwarmDefinition via closure."""
    from .swarm import AgentDefinition, TaskDefinition

    @tool
    def list_available_tools() -> str:
        """List all tools available for spawned agents."""
        tools = context.capabilities.available_tool_names
        return "Available tools:\n" + "\n".join(f"  - {t}" for t in tools) if tools else "No tools available."

    @tool
    def list_available_models() -> str:
        """List all models available for spawned agents."""
        models = context.capabilities.available_model_names
        return "Available models:\n" + "\n".join(f"  - {m}" for m in models) if models else "No models configured."

    @tool
    def spawn_agent(
        name: str,
        role: str,
        instructions: str = "",
        tools: list[str] | None = None,
        model: str | None = None,
    ) -> str:
        """Spawn a new sub-agent with specific capabilities."""
        definition = AgentDefinition(
            name=name,
            role=role,
            instructions=instructions or None,
            tools=tools or [],
            model=model,
        )

        try:
            context.register_agent(definition)
            agent_def = context.sub_agents.get(name)
            context.emit(AgentSpawnedEvent(
                name=name,
                role=role,
                instructions=instructions or None,
                tools=tools or [],
                model=model,
                color=agent_def.color if agent_def else None,
            ))
            return f"✓ Spawned '{name}' ({role})"
        except ValueError as e:
            return f"✗ Failed: {e}"

    @tool
    def create_task(
        name: str,
        agent_name: str,
        description: str = "",
        depends_on: list[str] | None = None,
    ) -> str:
        """Create a task and assign it to a spawned agent."""
        definition = TaskDefinition(
            name=name,
            agent=agent_name,
            description=description or None,
            depends_on=depends_on or [],
        )

        try:
            context.register_task(definition)
            context.emit(TaskCreatedEvent(
                name=name,
                agent=agent_name,
                description=description or None,
                depends_on=depends_on or [],
            ))
            return f"✓ Created '{name}' -> '{agent_name}'"
        except ValueError as e:
            return f"✗ Failed: {e}"

    @tool
    def get_planning_status() -> str:
        """Get current planning status - agents and tasks defined so far."""
        return context.get_summary()

    @tool
    def finalize_plan() -> str:
        """Signal that planning is complete. Call after creating all agents and tasks."""
        context.emit(PlanningCompletedEvent(
            entry_task=None,
            agents=list(context.sub_agents.values()),
            tasks=list(context.tasks.values()),
        ))
        return f"✓ Planning complete.\n\n{context.get_summary()}"

    return [
        list_available_tools,
        list_available_models,
        spawn_agent,
        create_task,
        get_planning_status,
        finalize_plan,
    ]


def create_orchestrator_agent(
    context: "SwarmDefinition",
    model: "Model | None" = None,
) -> Agent:
    """Create orchestrator agent that plans, creates sub-agents, assigns tasks, and synthesizes completion."""
    tools = create_orchestrator_tools(context)
    return Agent(
        name="orchestrator",
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        model=model,
        tools=tools,
    )
