"""Example: DynamicSwarm - Auto-built multi-agent workflows.

This example demonstrates how DynamicSwarm automatically:
1. Analyzes your query
2. Spawns specialized sub-agents with appropriate tools and models
3. Creates tasks with dependencies
4. Executes the workflow

No pre-defined agents needed - the planner creates them dynamically!

Current version: Rollout-only (string-in, string-out execution).
RL support coming soon via strands-sglang integration.
"""

from strands import tool
from strands.models import BedrockModel
from strands_swarms import (
    DynamicSwarm,
    HookProvider,
    HookRegistry,
    AgentSpawnedEvent,
    TaskCreatedEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
    SwarmCompletedEvent,
    SwarmFailedEvent,
)


# =============================================================================
# Define tools that spawned agents can use
# =============================================================================


@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query.

    Returns:
        Search results as text.
    """
    # In a real implementation, this would call a search API
    return f"[Search Results for '{query}']\n- Result 1: Latest developments...\n- Result 2: Key trends..."


@tool
def analyze_data(data: str) -> str:
    """Analyze data and extract insights.

    Args:
        data: The data to analyze.

    Returns:
        Analysis results.
    """
    return f"[Analysis]\nKey insights from the data:\n1. Trend identified\n2. Pattern detected\n3. Recommendation: ..."


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file.

    Args:
        path: File path to write to.
        content: Content to write.

    Returns:
        Confirmation message.
    """
    # In a real implementation, this would write to disk
    print(f"\n--- Writing to {path} ---\n{content[:200]}...\n")
    return f"Successfully wrote {len(content)} characters to {path}"


@tool
def execute_code(code: str) -> str:
    """Execute Python code safely.

    Args:
        code: Python code to execute.

    Returns:
        Execution output.
    """
    # In a real implementation, this would use a sandbox
    return f"[Code Output]\nExecuted successfully. Output: ..."


# =============================================================================
# Custom Hook Provider Example (strands-compatible)
# =============================================================================


class TimestampedHookProvider(HookProvider):
    """Example hook provider that logs events with timestamps.
    
    This follows the strands HookProvider pattern for type-safe event handling.
    """
    
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        """Register callbacks for events we care about."""
        registry.add_callback(AgentSpawnedEvent, self._on_agent_spawned)
        registry.add_callback(TaskCreatedEvent, self._on_task_created)
        registry.add_callback(TaskStartedEvent, self._on_task_started)
        registry.add_callback(TaskCompletedEvent, self._on_task_completed)
        registry.add_callback(SwarmCompletedEvent, self._on_swarm_completed)
        registry.add_callback(SwarmFailedEvent, self._on_swarm_failed)
    
    def _timestamp(self) -> str:
        import time
        return time.strftime("%H:%M:%S")
    
    def _on_agent_spawned(self, event: AgentSpawnedEvent) -> None:
        print(f"[{self._timestamp()}] 🤖 Agent '{event.name}' spawned with role: {event.role}")
    
    def _on_task_created(self, event: TaskCreatedEvent) -> None:
        deps = event.depends_on
        print(f"[{self._timestamp()}] 📋 Task '{event.name}' created" + (f" (depends on: {deps})" if deps else ""))
    
    def _on_task_started(self, event: TaskStartedEvent) -> None:
        print(f"[{self._timestamp()}] ▶️  Task '{event.name}' started")
    
    def _on_task_completed(self, event: TaskCompletedEvent) -> None:
        print(f"[{self._timestamp()}] ✓  Task '{event.name}' completed")
    
    def _on_swarm_completed(self, event: SwarmCompletedEvent) -> None:
        print(f"[{self._timestamp()}] 🏁 Swarm completed!")
    
    def _on_swarm_failed(self, event: SwarmFailedEvent) -> None:
        print(f"[{self._timestamp()}] ❌ Swarm failed: {event.error}")


# =============================================================================
# Create and run DynamicSwarm
# =============================================================================


def main():
    # Create Model instances from strands
    # You can use any strands Model: BedrockModel, AnthropicModel, LiteLLMModel, etc.
    powerful_model = BedrockModel(model_id="us.anthropic.claude-3-opus-20240229-v1:0")
    fast_model = BedrockModel(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0")

    # Create dynamic swarm with available tools and models
    swarm = DynamicSwarm(
        # Tools that spawned agents can use
        available_tools={
            "search_web": search_web,
            "analyze_data": analyze_data,
            "write_file": write_file,
            "execute_code": execute_code,
        },
        # Models that spawned agents can use
        # Keys are friendly names the planner uses, values are Model instances
        available_models={
            "powerful": powerful_model,
            "fast": fast_model,
        },
        # Model instance for the planner agent
        planner_model=powerful_model,
        # Default model name for spawned agents if not specified
        default_agent_model="fast",
        
        # Event handling options:
        # Option 1: verbose=True uses PrintingHookProvider (rich CLI output)
        verbose=True,
        
        # Option 2: Custom hook provider for your own logging/UI
        # hooks=[TimestampedHookProvider()],
    )

    # Example queries to try
    queries = [
        "Research the latest AI trends and write a summary report",
        "Analyze the competitive landscape in cloud computing",
        "Build a simple data pipeline to process CSV files",
    ]

    query = queries[0]
    print(f"Query: {query}\n")
    print("=" * 60)

    # Execute - the planner will:
    # 1. Analyze the query
    # 2. Decide what agents to spawn (researcher, writer, etc.)
    # 3. Assign tools and models to each agent
    # 4. Create tasks with proper dependencies
    # 5. Execute the workflow
    result = swarm.execute(query)

    # Inspect results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Status: {result.status}")
    print(f"Agents spawned: {result.agents_spawned}")
    print(f"Tasks created: {result.tasks_created}")
    print(f"Execution mode: {result.execution_mode}")

    if result.final_response:
        print(f"\nFinal response:\n{result.final_response}")


if __name__ == "__main__":
    main()
