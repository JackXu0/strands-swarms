"""Example: DynamicSwarm - Auto-built multi-agent workflows.

DynamicSwarm automatically analyzes your query, spawns specialized sub-agents
with appropriate tools and models, creates tasks with dependencies, and
executes the workflow. No pre-defined agents needed!

Current version: Rollout-only (string-in, string-out execution).
RL support coming soon via strands-sglang integration.
"""

import time

from strands import tool
from strands.models import BedrockModel

from strands_swarms import (
    AgentSpawnedEvent,
    DynamicSwarm,
    HookProvider,
    HookRegistry,
    SwarmCompletedEvent,
    SwarmFailedEvent,
    TaskCompletedEvent,
    TaskCreatedEvent,
    TaskStartedEvent,
)


# =============================================================================
# All Tools Available to Swarm
# =============================================================================

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[Search Results for '{query}']\n- Result 1: Latest developments...\n- Result 2: Key trends..."


@tool
def analyze_data(data: str) -> str:
    """Analyze data and extract insights."""
    return "[Analysis]\nKey insights:\n1. Trend identified\n2. Pattern detected\n3. Recommendation: ..."


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    print(f"\n--- Writing to {path} ---\n{content[:200]}...\n")
    return f"Successfully wrote {len(content)} characters to {path}"


@tool
def execute_code(code: str) -> str:
    """Execute Python code safely."""
    return "[Code Output]\nExecuted successfully. Output: ..."


TOOLS = {
    "search_web": search_web,
    "analyze_data": analyze_data,
    "write_file": write_file,
    "execute_code": execute_code,
}


# =============================================================================
# All Models Available to Swarm
# =============================================================================
# Strands supports: AnthropicModel, OpenAIModel, BedrockModel, GeminiModel,
# LiteLLMModel, OllamaModel, MistralModel, LlamaCppModel, SageMakerModel, etc.
# See: https://github.com/strands-agents/sdk-python/tree/main/src/strands/models

MODELS = {
    "powerful": BedrockModel(model_id="us.anthropic.claude-3-opus-20240229-v1:0"),
    "fast": BedrockModel(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0"),
}


# =============================================================================
# Main
# =============================================================================

def main():
    swarm = DynamicSwarm(
        available_tools=TOOLS,
        available_models=MODELS,
        orchestrator_model=MODELS["powerful"],
        default_agent_model="fast",
        verbose=True,  # or hooks=[TimestampedHookProvider()]
    )

    query = "Research the latest AI trends and write a summary report"
    print(f"Query: {query}\n{'=' * 60}")

    result = swarm.execute(query)

    print("\n" + "=" * 60)
    print(f"Status: {result.status}")
    print(f"Agents spawned: {result.agents_spawned}")
    print(f"Tasks created: {result.tasks_created}")
    if result.final_response:
        print(f"\nFinal response:\n{result.final_response}")


# =============================================================================
# Hooks
# =============================================================================

class TimestampedHookProvider(HookProvider):
    """Logs swarm events with timestamps."""

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AgentSpawnedEvent, self._on_agent_spawned)
        registry.add_callback(TaskCreatedEvent, self._on_task_created)
        registry.add_callback(TaskStartedEvent, self._on_task_started)
        registry.add_callback(TaskCompletedEvent, self._on_task_completed)
        registry.add_callback(SwarmCompletedEvent, self._on_swarm_completed)
        registry.add_callback(SwarmFailedEvent, self._on_swarm_failed)

    def _ts(self) -> str:
        return time.strftime("%H:%M:%S")

    def _on_agent_spawned(self, event: AgentSpawnedEvent) -> None:
        print(f"[{self._ts()}] 🤖 Agent '{event.name}' spawned: {event.role}")

    def _on_task_created(self, event: TaskCreatedEvent) -> None:
        deps = f" (depends on: {event.depends_on})" if event.depends_on else ""
        print(f"[{self._ts()}] 📋 Task '{event.name}' created{deps}")

    def _on_task_started(self, event: TaskStartedEvent) -> None:
        print(f"[{self._ts()}] ▶️  Task '{event.name}' started")

    def _on_task_completed(self, event: TaskCompletedEvent) -> None:
        print(f"[{self._ts()}] ✓  Task '{event.name}' completed")

    def _on_swarm_completed(self, event: SwarmCompletedEvent) -> None:
        print(f"[{self._ts()}] 🏁 Swarm completed!")

    def _on_swarm_failed(self, event: SwarmFailedEvent) -> None:
        print(f"[{self._ts()}] ❌ Swarm failed: {event.error}")


if __name__ == "__main__":
    main()
