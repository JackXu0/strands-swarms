"""Example: DynamicSwarm - Auto-built multi-agent workflows.

DynamicSwarm automatically analyzes your query, spawns specialized sub-agents
with appropriate tools and models, creates tasks with dependencies, and
executes the workflow. No pre-defined agents needed!

Current version: Rollout-only (string-in, string-out execution).
RL support coming soon via strands-sglang integration.
"""

import asyncio

from strands import tool
from strands.models import BedrockModel

from strands_swarms import DynamicSwarm, TaskDefinition


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

def _format_status(status: object) -> str:
    value = getattr(status, "value", None)
    return str(value) if value is not None else str(status)


def _render_dashboard(
    tasks: dict[str, TaskDefinition],
    status_by_task: dict[str, str],
) -> str:
    def _blocked_by(task_name: str) -> list[str]:
        deps = tasks[task_name].depends_on
        return [d for d in deps if status_by_task.get(d) != "completed"]

    agents = sorted({t.agent for t in tasks.values()})
    lines = ["\n" + "-" * 40, "TASK STATUS", "-" * 40]

    for agent in agents:
        task_names = sorted([n for n, t in tasks.items() if t.agent == agent])
        if not task_names:
            continue

        rendered: list[str] = []
        for name in task_names:
            status = status_by_task.get(name, "unknown")
            if status == "pending":
                blocked = _blocked_by(name)
                if blocked:
                    status = f"pending (blocked: {', '.join(blocked)})"
                else:
                    status = "pending (ready)"
            rendered.append(f"{name} [{status}]")

        lines.append(f"{agent}: " + ", ".join(rendered))

    return "\n".join(lines) + "\n"


async def main() -> None:
    swarm = DynamicSwarm(
        available_tools=TOOLS,
        available_models=MODELS,
        orchestrator_model=MODELS["powerful"],
        default_agent_model="fast",
    )

    query = "Research the latest AI trends and write a summary report"
    print(f"Query: {query}\n{'=' * 60}\n")

    result = None
    tasks: dict[str, TaskDefinition] = {}
    status_by_task: dict[str, str] = {}
    async for event in swarm.stream_async(query):
        t = event.get("type")

        if t == "swarm_started":
            print("=" * 60)
            print("DYNAMIC SWARM STARTING")
            print("=" * 60)
            print(f"\nQuery: {event.get('query')}")
            print(f"Available tools: {event.get('available_tools') or ['none']}")
            print(f"Available models: {event.get('available_models') or ['default']}")

        elif t == "planning_started":
            print("\n" + "-" * 40)
            print("PHASE 1: PLANNING")
            print("-" * 40)

        elif t == "planning_completed":
            print("\n" + "·" * 40)
            print("PLAN READY")
            print("·" * 40)
            print(event.get("summary", ""))
            tasks = event.get("tasks") or {}

        elif t == "execution_started":
            print("\n" + "-" * 40)
            print("PHASE 2: EXECUTION")
            print("-" * 40)
            task_names = event.get("tasks") or sorted(tasks.keys())
            status_by_task = {name: "pending" for name in task_names}
            print(_render_dashboard(tasks, status_by_task))

        elif t == "multiagent_node_start":
            node_id = event.get("node_id")
            if node_id in status_by_task:
                status_by_task[node_id] = "executing"
                print(_render_dashboard(tasks, status_by_task))

        elif t == "multiagent_node_stop":
            node_id = event.get("node_id")
            node_result = event.get("node_result")
            if node_id in status_by_task:
                status_by_task[node_id] = _format_status(
                    getattr(node_result, "status", None)
                )
                print(_render_dashboard(tasks, status_by_task))

        elif t == "multiagent_result":
            graph_result = event.get("result")
            status = getattr(graph_result, "status", None)
            print("\n" + "-" * 40)
            print("EXECUTION COMPLETE")
            print("-" * 40)
            print(f"Status: {_format_status(status)}")

        elif t == "synthesis_completed":
            final = event.get("final_response")
            if final:
                print("\nFinal response:\n" + str(final))

        elif t == "swarm_result":
            result = event.get("result")

    if result is not None:
        print("\n" + "=" * 60)
        print(f"Status: {result.status}")
        print(f"Agents spawned: {result.agents_spawned}")
        print(f"Tasks created: {result.tasks_created}")


if __name__ == "__main__":
    asyncio.run(main())
