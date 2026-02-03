"""Orchestrator agent that plans and creates sub-agents and tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from strands import Agent, tool

from .events import AgentSpawnedEvent, PlanningCompletedEvent, TaskCreatedEvent

if TYPE_CHECKING:
    from strands.models import Model

    from .definition import SwarmDefinition


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a workflow orchestrator. Your job is to break down requests into sub-agents and tasks.

## Tools

- **spawn_agent(name, role, tools)** - Create a specialized agent
- **create_task(name, agent_name, description, depends_on)** - Create a task for an agent
- **finalize_plan()** - Call when done planning

## Process

1. Analyze what needs to be done
2. Create agents with the tools they need
3. Create tasks, specifying dependencies if one task needs another's output
   - Important: dependencies must already exist (create dependency tasks first)
4. Call finalize_plan()

## Example

Request: "Research AI trends and write a summary"

1. spawn_agent(name="researcher", role="research specialist", tools=["search_web"])
2. spawn_agent(name="writer", role="technical writer", tools=[])
3. create_task(name="research", agent_name="researcher", description="Research AI trends")
4. create_task(name="write", agent_name="writer", description="Write summary", depends_on=["research"])
5. finalize_plan()

Keep it simple - only create what's necessary."""


def create_orchestrator_tools(definition: "SwarmDefinition") -> list[Callable[..., str]]:
    """Create orchestrator tools that capture the SwarmDefinition via closure."""
    from .definition import AgentDefinition, TaskDefinition

    @tool
    def spawn_agent(
        name: str,
        role: str,
        tools: list[str] | None = None,
        instructions: str = "",
        model: str | None = None,
    ) -> str:
        """Create a sub-agent with specific capabilities."""
        agent_def = AgentDefinition(
            name=name,
            role=role,
            instructions=instructions or None,
            tools=tools or [],
            model=model,
        )

        try:
            definition.register_agent(agent_def)
            registered = definition.sub_agents.get(name)
            definition.emit(AgentSpawnedEvent(
                name=name,
                role=role,
                instructions=instructions or None,
                tools=tools or [],
                model=model,
                color=registered.color if registered else None,
            ))
            return f"Created agent '{name}' ({role})"
        except ValueError as e:
            return f"Error: {e}"

    @tool
    def create_task(
        name: str,
        agent_name: str,
        description: str = "",
        depends_on: list[str] | None = None,
    ) -> str:
        """Create a task assigned to a sub-agent.

        Note: any tasks listed in depends_on must already exist (create dependency tasks first).
        """
        task_def = TaskDefinition(
            name=name,
            agent=agent_name,
            description=description or None,
            depends_on=depends_on or [],
        )

        try:
            definition.register_task(task_def)
            definition.emit(TaskCreatedEvent(
                name=name,
                agent=agent_name,
                description=description or None,
                depends_on=depends_on or [],
            ))
            deps = f" (after: {depends_on})" if depends_on else ""
            return f"Created task '{name}' -> {agent_name}{deps}"
        except ValueError as e:
            return f"Error: {e}"

    @tool
    def finalize_plan() -> str:
        """Signal that planning is complete."""
        definition.emit(PlanningCompletedEvent(
            entry_task=None,
            agents=list(definition.sub_agents.values()),
            tasks=list(definition.tasks.values()),
        ))
        return f"Plan finalized.\n{definition.get_summary()}"

    return [spawn_agent, create_task, finalize_plan]


def create_orchestrator_agent(
    definition: "SwarmDefinition",
    model: "Model | None" = None,
) -> Agent:
    """Create the orchestrator agent."""
    return Agent(
        name="orchestrator",
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        model=model,
        tools=create_orchestrator_tools(definition),
    )
