"""DynamicSwarm - automatically construct and execute multi-agent workflows.

This module provides the core swarm functionality:
- Agent and task definitions
- SwarmConfig for collecting definitions and building the swarm graph
- DynamicSwarm class that uses an orchestrator for:
  1. Planning and creating subagents
  2. Assigning tasks
  3. Generating final response
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable

from strands import Agent
from strands.hooks import HookProvider, HookRegistry
from strands.multiagent.base import MultiAgentResult, Status
from strands.multiagent.graph import Graph, GraphBuilder, GraphResult

from .events import (
    AGENT_COLORS,
    ExecutionCompletedEvent,
    ExecutionStartedEvent,
    PlanningStartedEvent,
    PrintingHookProvider,
    SwarmCompletedEvent,
    SwarmFailedEvent,
    SwarmStartedEvent,
    create_colored_callback_handler,
)

if TYPE_CHECKING:
    from strands.models import Model
    from strands.session import SessionManager


# =============================================================================
# Agent and Task Definitions
# =============================================================================


@dataclass
class AgentDefinition:
    """Definition for a dynamically spawned agent.

    Attributes:
        name: Unique identifier for this agent.
        role: What this agent does (used in system prompt).
        instructions: Additional instructions for the agent.
        tools: List of tool names from the available pool.
        model: Model name from the available pool.
        color: Display color for this agent (auto-assigned on registration).
    """

    name: str
    role: str
    instructions: str | None = None
    tools: list[str] = field(default_factory=list)
    model: str | None = None
    color: str | None = None

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


# =============================================================================
# Swarm Configuration
# =============================================================================


class SwarmConfig:
    """Configuration for a dynamic swarm.

    Collects agent and task definitions during planning. Call build() to
    create a complete Graph ready for execution.
    """

    def __init__(
        self,
        available_tools: dict[str, Callable[..., Any]],
        available_models: dict[str, Model],
        default_model: str | None = None,
    ) -> None:
        """Initialize the swarm configuration.

        Args:
            available_tools: Pool of tools that can be assigned to agents.
            available_models: Pool of models (name -> Model instance) that can be used by agents.
            default_model: Default model name to use if none specified.
        """
        self._available_tools = available_tools
        self._available_models = available_models
        self._default_model = default_model or next(iter(available_models.keys()), None)

        self._agents: dict[str, AgentDefinition] = {}
        self._tasks: dict[str, TaskDefinition] = {}
        self._entry_task: str | None = None
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

        # Assign a color to this agent (used by events for display)
        definition.color = AGENT_COLORS[self._color_index % len(AGENT_COLORS)]
        self._color_index += 1
        
        self._agents[definition.name] = definition

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

    def clear(self) -> None:
        """Clear all registered agents and tasks."""
        self._agents.clear()
        self._tasks.clear()
        self._color_index = 0
        self._entry_task = None

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
    def available_tools(self) -> dict[str, Callable[..., Any]]:
        """Get available tools mapping."""
        return self._available_tools

    @property
    def available_models(self) -> dict[str, Model]:
        """Get available models mapping."""
        return self._available_models

    @property
    def default_model(self) -> str | None:
        """Get the default model name."""
        return self._default_model

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
        return {name: d.color for name, d in self._agents.items() if d.color}
    
    def get_summary(self) -> str:
        """Get a summary of the current swarm configuration."""
        lines = [
            f"Agents ({len(self._agents)}):",
            *[f"  - {name}: {d.role}" for name, d in self._agents.items()],
            f"\nTasks ({len(self._tasks)}):",
            *[
                f"  - {name} -> {d.agent}" + (f" (depends: {d.depends_on})" if d.depends_on else "")
                for name, d in self._tasks.items()
            ],
            f"\nEntry: {self._entry_task or 'auto'}",
        ]
        return "\n".join(lines)


@dataclass
class _SwarmBuildResult:
    """Result from building a swarm, including the graph and agent tracking."""

    graph: Graph
    task_agents: dict[str, Agent]  # task_name -> Agent instance


def build_swarm(
    config: SwarmConfig,
    *,
    use_colored_output: bool = False,
    execution_timeout: float = 900.0,
    task_timeout: float = 300.0,
    session_manager: SessionManager | None = None,
) -> _SwarmBuildResult:
    """Build a complete Graph from a swarm configuration.

    Creates all agents from registered definitions and wires up the
    task dependency graph.

    Args:
        config: The swarm configuration containing agent and task definitions.
        use_colored_output: Whether to use colored output for agents.
        execution_timeout: Overall execution timeout in seconds.
        task_timeout: Per-task timeout in seconds.
        session_manager: Optional session manager for persistence.

    Returns:
        _SwarmBuildResult containing the graph and agent/model tracking.

    Raises:
        ValueError: If no tasks are registered.
    """
    task_agents: dict[str, Agent] = {}

    def build_agent(agent_name: str, task_name: str) -> Agent:
        """Build an Agent from a registered definition."""
        definition = config.agents.get(agent_name)
        if not definition:
            raise ValueError(f"Agent '{agent_name}' not found")

        task_def = config.tasks.get(task_name)

        tools = [config.available_tools[t] for t in definition.tools]
        model_name = definition.model or config.default_model
        model = config.available_models.get(model_name) if model_name else None

        callback_handler = None
        if use_colored_output and definition.color:
            callback_handler = create_colored_callback_handler(definition.color, agent_name)

        # Build system prompt with task description included
        system_prompt = definition.build_system_prompt()
        if task_def and task_def.description:
            system_prompt += f"\n\n## Your Task\n{task_def.description}"

        agent = Agent(
            name=definition.name,
            system_prompt=system_prompt,
            model=model,
            tools=tools if tools else None,  # type: ignore[arg-type]
            callback_handler=callback_handler,
        )
        task_agents[task_name] = agent
        return agent

    if not config.tasks:
        raise ValueError("No tasks registered - cannot build swarm")

    builder = GraphBuilder()

    # Build agents and add as nodes
    for task_name, task_def in config.tasks.items():
        builder.add_node(build_agent(task_def.agent, task_name), task_name)

    # Add edges based on dependencies
    for task_name, task_def in config.tasks.items():
        for dep_name in task_def.depends_on:
            if dep_name in config.tasks:
                builder.add_edge(dep_name, task_name)

    # Set entry point
    if config.entry_task:
        builder.set_entry_point(config.entry_task)

    # Configure execution
    builder.set_execution_timeout(execution_timeout)
    builder.set_node_timeout(task_timeout)

    if session_manager:
        builder.set_session_manager(session_manager)

    return _SwarmBuildResult(
        graph=builder.build(),
        task_agents=task_agents,
    )


# =============================================================================
# DynamicSwarm Result
# =============================================================================


@dataclass
class DynamicSwarmResult:
    """Result from DynamicSwarm execution.

    Attributes:
        status: Overall execution status.
        planning_output: Output from the planning phase.
        execution_result: Result from the execution phase (GraphResult).
        final_response: Final response from the planner after all tasks completed.
        agents_spawned: Number of agents that were dynamically spawned.
        tasks_created: Number of tasks that were created.
    """

    status: Status
    planning_output: str | None = None
    execution_result: GraphResult | None = None
    final_response: str | None = None
    agents_spawned: int = 0
    tasks_created: int = 0
    error: str | None = None

    def get_output(self, task_name: str) -> Any | None:
        """Get output from a specific task."""
        if self.execution_result and hasattr(self.execution_result, "results"):
            node_result = self.execution_result.results.get(task_name)
            if node_result:
                return str(node_result.result)
        return None

    def __bool__(self) -> bool:
        """Return True if completed successfully."""
        return self.status == Status.COMPLETED


# =============================================================================
# DynamicSwarm Orchestrator
# =============================================================================


class DynamicSwarm:
    """Dynamically construct and execute multi-agent workflows.

    DynamicSwarm uses an orchestrator agent to analyze user queries and coordinate
    a multi-agent workflow. The orchestrator has three main responsibilities:

    1. **Planning and Creating Subagents** - Analyze the task and spawn specialized
       agents with appropriate tools and models.
    
    2. **Assigning Tasks** - Create tasks and assign them to the spawned agents,
       defining dependencies between tasks when needed.
    
    3. **Generating Final Response** - After all tasks complete, synthesize the
       results into a cohesive final response.

    All sub-agents run in parallel unless they have dependencies declared.
    Tasks with dependencies will wait for their dependencies to complete first.

    This is the unique value of this package - dynamic LLM-driven workflow orchestration.
    For static multi-agent workflows, use the Strands SDK directly:
    - strands.multiagent.graph.Graph for dependency-based execution

    Example:
        from strands_swarms import DynamicSwarm
        from strands import tool

        @tool
        def search_web(query: str) -> str:
            '''Search the web.'''
            return f"Results for: {query}"

        swarm = DynamicSwarm(
            available_tools={"search_web": search_web},
            verbose=True,
        )

        result = swarm.execute("Research AI trends and summarize")

    Customization:
        Prompt templates can be overridden via subclassing:

        class CustomSwarm(DynamicSwarm):
            ORCHESTRATION_PROMPT_TEMPLATE = "Your custom template with {query}, {available_tools}, {available_models}"
    """

    ORCHESTRATION_PROMPT_TEMPLATE: str = dedent("""\
        Analyze this request and design a workflow:

        {query}

        Available tools: {available_tools}
        Available models: {available_models}

        Create the necessary agents and tasks, then call execute_swarm() when ready.""")

    COMPLETION_PROMPT_TEMPLATE: str = dedent("""\
        The tasks you designed have completed. Here are the outputs from each agent:

        {task_outputs}

        Now synthesize these results into a final, cohesive response to the original query:
        {query}
        
        Be direct and deliver the result. Don't explain the process or mention the sub-agents.""")

    def __init__(
        self,
        available_tools: dict[str, Callable[..., Any]] | None = None,
        available_models: dict[str, Model] | None = None,
        *,
        orchestrator_model: Model | None = None,
        default_agent_model: str | None = None,
        max_iterations: int = 20,
        execution_timeout: float = 900.0,
        task_timeout: float = 300.0,
        session_manager: SessionManager | None = None,
        hooks: list[HookProvider] | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize DynamicSwarm.

        Args:
            available_tools: Pool of tools that can be assigned to spawned agents.
                            Keys are tool names used by the orchestrator.
            available_models: Pool of Model instances that can be used by spawned agents.
                             Keys are friendly names (e.g., "fast", "powerful"),
                             values are Model instances.
            orchestrator_model: Model instance to use for the orchestrator agent.
            default_agent_model: Default model name (key in available_models) for spawned agents.
            max_iterations: Maximum iterations for execution.
            execution_timeout: Overall execution timeout in seconds.
            task_timeout: Per-task timeout in seconds.
            session_manager: Optional session manager for persistence.
            hooks: List of HookProvider instances for event callbacks.
                  Use PrintingHookProvider() for CLI output.
            verbose: Shorthand for hooks=[PrintingHookProvider()].
        """
        self._available_tools = available_tools or {}
        self._available_models = available_models or {}
        self._orchestrator_model = orchestrator_model
        self._default_agent_model = default_agent_model
        self._max_iterations = max_iterations
        self._execution_timeout = execution_timeout
        self._task_timeout = task_timeout
        self._session_manager = session_manager
        
        # Build hook registry
        self._hook_registry = HookRegistry()
        
        # Add verbose hook if requested
        if verbose:
            self._hook_registry.add_hook(PrintingHookProvider())
        
        # Add user-provided hooks
        if hooks:
            for hook in hooks:
                self._hook_registry.add_hook(hook)

    def execute(self, query: str) -> DynamicSwarmResult:
        """Execute a query by dynamically building and running a workflow.

        Args:
            query: The user's request to process.

        Returns:
            DynamicSwarmResult containing planning and execution results.
        """
        return asyncio.get_event_loop().run_until_complete(self.execute_async(query))

    def _emit(self, event: Any) -> None:
        """Emit an event to all registered hooks."""
        if self._hook_registry.has_callbacks():
            self._hook_registry.invoke_callbacks(event)

    async def execute_async(self, query: str) -> DynamicSwarmResult:
        """Execute asynchronously.

        Args:
            query: The user's request to process.

        Returns:
            DynamicSwarmResult containing planning and execution results.
        """
        # Import here to avoid circular import
        from .orchestrator import set_swarm_config

        # Create swarm config for this execution
        config = SwarmConfig(
            available_tools=self._available_tools,
            available_models=self._available_models,
            default_model=self._default_agent_model,
        )

        # Set config for planning tools (with hook registry)
        set_swarm_config(config, hook_registry=self._hook_registry)

        try:
            # Emit swarm started event
            self._emit(SwarmStartedEvent(
                query=query,
                available_tools=list(self._available_tools.keys()),
                available_models=list(self._available_models.keys()),
            ))
            
            # =================================================================
            # Orchestrator Phase 1 & 2: Planning, Creating Subagents, Assigning Tasks
            # =================================================================
            self._emit(PlanningStartedEvent())
            
            planning_result = await self._run_planning(query, config)
            if not planning_result.success:
                self._emit(SwarmFailedEvent(
                    error=planning_result.error or "Orchestration failed"
                ))
                return DynamicSwarmResult(
                    status=Status.FAILED,
                    planning_output=planning_result.output,
                    error=planning_result.error,
                )

            # Validate we have tasks
            if not config.tasks:
                self._emit(SwarmFailedEvent(
                    error="Orchestration completed but no tasks were created"
                ))
                return DynamicSwarmResult(
                    status=Status.FAILED,
                    planning_output=planning_result.output,
                    error="Orchestration completed but no tasks were created",
                )

            # Build the swarm graph (create sub-agents, wire up task dependencies)
            use_colored_output = self._hook_registry.has_callbacks()
            build_result = build_swarm(
                config,
                use_colored_output=use_colored_output,
                execution_timeout=self._execution_timeout,
                task_timeout=self._task_timeout,
                session_manager=self._session_manager,
            )

            # =================================================================
            # Execute Tasks
            # =================================================================
            self._emit(ExecutionStartedEvent(
                tasks=list(config.tasks.keys()),
            ))
            
            execution_result = await build_result.graph.invoke_async()

            # Emit execution completion event
            self._emit(ExecutionCompletedEvent(
                status=str(execution_result.status) if execution_result else "FAILED",
                agent_count=len(config.agents),
                task_count=len(config.tasks),
            ))

            # =================================================================
            # Orchestrator Phase 3: Generate Final Response
            # Uses the SAME orchestrator agent from planning (continued conversation)
            # =================================================================
            assert planning_result.orchestrator is not None, "Orchestrator should exist after successful planning"
            final_response = await self._run_completion(
                query, config, execution_result, 
                orchestrator=planning_result.orchestrator
            )
            
            self._emit(SwarmCompletedEvent())

            return DynamicSwarmResult(
                status=execution_result.status if execution_result else Status.FAILED,
                planning_output=planning_result.output,
                execution_result=execution_result,
                final_response=final_response,
                agents_spawned=len(config.agents),
                tasks_created=len(config.tasks),
            )

        finally:
            # Clear config
            set_swarm_config(None)

    async def _run_planning(
        self, query: str, state: SwarmConfig
    ) -> _PlanningResult:
        """Run the orchestration phase: planning and creating subagents, assigning tasks.
        
        Returns the orchestrator agent along with the result so it can be reused
        for the completion phase (same agent, continued conversation).
        """
        from .orchestrator import create_orchestrator_agent

        orchestrator = create_orchestrator_agent(model=self._orchestrator_model)

        orchestration_prompt = self.ORCHESTRATION_PROMPT_TEMPLATE.format(
            query=query,
            available_tools=state.available_tool_names or ['none'],
            available_models=state.available_model_names or ['default only'],
        )

        try:
            result = orchestrator(orchestration_prompt)

            output = None
            if hasattr(result, "message") and result.message:
                content = result.message.get("content", [])
                if content and isinstance(content[0], dict):
                    output = content[0].get("text", "")

            return _PlanningResult(success=True, output=output, orchestrator=orchestrator)

        except Exception as e:
            return _PlanningResult(success=False, error=str(e))

    async def _run_completion(
        self,
        query: str,
        state: SwarmConfig,
        execution_result: MultiAgentResult | None,
        orchestrator: Agent,
    ) -> str | None:
        """Run the final response generation phase - synthesize task outputs into a cohesive response.
        
        This is the third responsibility of the orchestrator: generating the final response
        by combining outputs from all sub-agents.
        
        Uses the SAME orchestrator agent that did the planning, continuing the conversation
        so it has full context about why agents were spawned and what tasks were meant to do.
        
        Args:
            query: Original user query.
            state: Swarm configuration with task definitions.
            execution_result: Results from task execution.
            orchestrator: The orchestrator agent from planning phase (same agent, continued conversation).
        """
        if not execution_result or not hasattr(execution_result, "results"):
            return None

        # Collect all task outputs
        task_outputs: list[str] = []
        for task_name in state.tasks:
            node_result = execution_result.results.get(task_name)
            if node_result:
                task_outputs.append(f"[{task_name}]:\n{node_result.result}")

        if not task_outputs:
            return None

        completion_prompt = self.COMPLETION_PROMPT_TEMPLATE.format(
            query=query,
            task_outputs="\n\n".join(task_outputs),
        )
        try:
            result = orchestrator(completion_prompt)
            
            if hasattr(result, "message") and result.message:
                content = result.message.get("content", [])
                if content and isinstance(content[0], dict):
                    return content[0].get("text", "")
            return None
        except Exception:
            return None

@dataclass
class _PlanningResult:
    """Internal result from planning phase."""

    success: bool
    output: str | None = None
    error: str | None = None
    orchestrator: Agent | None = None  # Preserved for completion phase

