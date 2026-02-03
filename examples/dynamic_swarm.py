"""Example: DynamicSwarm - Auto-built multi-agent workflows.

DynamicSwarm automatically analyzes your query, spawns specialized sub-agents
with appropriate tools and models, creates tasks with dependencies, and
executes the workflow. No pre-defined agents needed!

Current version: Rollout-only (string-in, string-out execution).
RL support coming soon via strands-sglang integration.
"""

import asyncio
import os
import sys

from strands import tool
from strands.models import BedrockModel

from strands_swarms import AgentDefinition, DynamicSwarm, TaskDefinition


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

RESET = "\033[0m"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    # Example script: default to color for readability, even if stdout isn't a TTY.
    return True


def _colorize(text: str, color: str | None, *, enabled: bool) -> str:
    if not enabled or not color:
        return text
    return f"{color}{text}{RESET}"


def _format_status(status: object) -> str:
    value = getattr(status, "value", None)
    return str(value) if value is not None else str(status)


def _extract_text(result: object) -> str:
    message = getattr(result, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list) and content:
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            if parts:
                return "".join(parts)
    return str(result)


def _render_plan(
    agents: dict[str, AgentDefinition],
    tasks: dict[str, TaskDefinition],
    *,
    color_enabled: bool,
) -> str:
    task_to_agent = {name: t.agent for name, t in tasks.items()}

    def _agent_label(name: str) -> str:
        return _colorize(
            name,
            getattr(agents.get(name), "color", None),
            enabled=color_enabled,
        )

    def _task_label(name: str) -> str:
        agent_name = task_to_agent.get(name)
        return _colorize(
            name,
            getattr(agents.get(agent_name), "color", None),
            enabled=color_enabled,
        )

    lines = [f"Agents ({len(agents)}):"]
    for name in sorted(agents.keys()):
        lines.append(f"  - {_agent_label(name)}: {agents[name].role}")

    lines.append(f"\nTasks ({len(tasks)}):")
    for task_name in sorted(tasks.keys()):
        task = tasks[task_name]
        depends = ""
        if task.depends_on:
            deps = ", ".join(_task_label(d) for d in task.depends_on)
            depends = f" (depends: [{deps}])"

        lines.append(
            f"  - {_task_label(task_name)} -> {_agent_label(task.agent)}{depends}"
        )

    return "\n".join(lines)


def _render_dashboard(
    agents: dict[str, AgentDefinition],
    tasks: dict[str, TaskDefinition],
    status_by_task: dict[str, str],
    *,
    color_enabled: bool,
) -> str:
    task_to_agent = {name: t.agent for name, t in tasks.items()}

    def _agent_label(name: str) -> str:
        return _colorize(
            name,
            getattr(agents.get(name), "color", None),
            enabled=color_enabled,
        )

    def _task_label(name: str) -> str:
        agent_name = task_to_agent.get(name)
        return _colorize(
            name,
            getattr(agents.get(agent_name), "color", None),
            enabled=color_enabled,
        )

    def _status_label(task_name: str, status: str) -> str:
        agent_name = task_to_agent.get(task_name)
        return _colorize(
            status,
            getattr(agents.get(agent_name), "color", None),
            enabled=color_enabled,
        )

    def _blocked_by(task_name: str) -> list[str]:
        deps = tasks[task_name].depends_on
        return [d for d in deps if status_by_task.get(d) != "completed"]

    agent_names = sorted({t.agent for t in tasks.values()})
    lines = ["\n" + "-" * 40, "TASK STATUS", "-" * 40]

    for agent in agent_names:
        task_names = sorted([n for n, t in tasks.items() if t.agent == agent])
        if not task_names:
            continue

        rendered: list[str] = []
        for name in task_names:
            status = status_by_task.get(name, "unknown")
            if status == "pending":
                blocked = _blocked_by(name)
                if blocked:
                    blocked_list = ", ".join(_task_label(d) for d in blocked)
                    status = f"pending (blocked: {blocked_list})"
                else:
                    status = "pending (ready)"
            if status in {"pending (ready)", "executing", "completed", "failed", "interrupted"}:
                status = _status_label(name, status)
            rendered.append(f"{_task_label(name)} [{status}]")

        lines.append(f"{_agent_label(agent)}: " + ", ".join(rendered))

    return "\n".join(lines) + "\n"


async def main() -> None:
    color_enabled = _supports_color()

    swarm = DynamicSwarm(
        available_tools=TOOLS,
        available_models=MODELS,
        orchestrator_model=MODELS["powerful"],
        default_agent_model="fast",
    )

    query = "Research the latest AI trends and write a summary report"
    print(f"Query: {query}\n{'=' * 60}\n")

    result = None
    agents: dict[str, AgentDefinition] = {}
    tasks: dict[str, TaskDefinition] = {}
    status_by_task: dict[str, str] = {}
    tool_counts: dict[str, int] = {}
    seen_tool_use_ids: dict[str, set[str]] = {}

    async for event in swarm.stream_async(query, include_subagent_events=True):
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
            agents = event.get("agents") or {}
            tasks = event.get("tasks") or {}
            print(_render_plan(agents, tasks, color_enabled=color_enabled))

        elif t == "execution_started":
            print("\n" + "-" * 40)
            print("PHASE 2: EXECUTION")
            print("-" * 40)
            task_names = event.get("tasks") or sorted(tasks.keys())
            status_by_task = {name: "pending" for name in task_names}
            print(
                _render_dashboard(
                    agents, tasks, status_by_task, color_enabled=color_enabled
                )
            )

        elif t == "multiagent_node_start":
            node_id = event.get("node_id")
            if node_id in status_by_task:
                status_by_task[node_id] = "executing"
                tool_counts[node_id] = 0
                seen_tool_use_ids[node_id] = set()
                print(
                    _render_dashboard(
                        agents, tasks, status_by_task, color_enabled=color_enabled
                    )
                )

        elif t == "multiagent_node_stream":
            node_id = event.get("node_id")
            agent_event = event.get("event")
            if node_id in status_by_task and isinstance(agent_event, dict):
                task_def = tasks.get(node_id)
                if task_def and task_def.agent in agents:
                    agent_def = agents[task_def.agent]

                    chunk = agent_event.get("event")
                    if isinstance(chunk, dict):
                        tool_use = (
                            chunk.get("contentBlockStart", {})
                            .get("start", {})
                            .get("toolUse")
                        )
                        tool_use_id = (
                            tool_use.get("toolUseId")
                            if isinstance(tool_use, dict)
                            else None
                        )
                        tool_name = (
                            tool_use.get("name") if isinstance(tool_use, dict) else None
                        )

                        if isinstance(tool_use_id, str) and isinstance(tool_name, str):
                            if tool_use_id not in seen_tool_use_ids.get(node_id, set()):
                                seen_tool_use_ids.setdefault(node_id, set()).add(tool_use_id)
                                tool_counts[node_id] = tool_counts.get(node_id, 0) + 1
                                print(
                                    _colorize(
                                        f"{task_def.agent}: Tool #{tool_counts[node_id]}: {tool_name}",
                                        agent_def.color,
                                        enabled=color_enabled,
                                    )
                                )

        elif t == "multiagent_node_stop":
            node_id = event.get("node_id")
            node_result = event.get("node_result")
            if node_id in status_by_task:
                status_by_task[node_id] = _format_status(
                    getattr(node_result, "status", None)
                )

                task_def = tasks.get(node_id)
                if task_def and task_def.agent in agents:
                    agent_def = agents[task_def.agent]
                    output = getattr(node_result, "result", None)
                    if output is not None:
                        header = f"{task_def.agent}: {node_id}"
                        print(_colorize(header, agent_def.color, enabled=color_enabled))
                        print(
                            _colorize(
                                _extract_text(output).strip() + "\n",
                                agent_def.color,
                                enabled=color_enabled,
                            )
                        )

                print(
                    _render_dashboard(
                        agents, tasks, status_by_task, color_enabled=color_enabled
                    )
                )

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
