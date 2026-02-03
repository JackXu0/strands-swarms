# strands-swarms

[![CI](https://github.com/JackXu0/strands-swarms/actions/workflows/ci.yml/badge.svg)](https://github.com/JackXu0/strands-swarms/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/strands-swarms)](https://pypi.org/project/strands-swarms/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Turn natural language into multi-agent workflows — automatically.**

Give `DynamicSwarm` a query, and it automatically plans the workflow, spawns specialized agents, and executes tasks with dependencies. No manual graph configuration or agent wiring required.

```python
swarm = DynamicSwarm(available_tools={...}, available_models={...})
result = swarm.execute("Research AI trends and write a summary report")
# → Spawns researcher + writer agents, handles dependencies, returns final report
```

## Inspiration

This project is inspired by [Kimi K2.5's Agent Swarm](https://www.kimi.com/blog/kimi-k2-5.html) — where a trainable orchestrator dynamically creates and coordinates sub-agents without predefined roles or workflows. The goal is to build an open-source foundation for training orchestrators that can spin up agent swarms on the fly.

## How It Works

![Architecture](assets/architecture.png)

## Installation

```bash
pip install strands-swarms
```

## Quick Start

```python
from strands import tool
from strands.models import BedrockModel
from strands_swarms import DynamicSwarm

# Define your tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[Search Results for '{query}']\n- Result 1: Latest developments..."

@tool
def analyze_data(data: str) -> str:
    """Analyze data and extract insights."""
    return f"[Analysis]\nKey insights: ..."

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return f"Successfully wrote {len(content)} characters to {path}"

# Configure models
powerful_model = BedrockModel(model_id="us.anthropic.claude-3-opus-20240229-v1:0")
fast_model = BedrockModel(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0")

# Create the swarm
swarm = DynamicSwarm(
    available_tools={
        "search_web": search_web,
        "analyze_data": analyze_data,
        "write_file": write_file,
    },
    available_models={
        "powerful": powerful_model,
        "fast": fast_model,
    },
    orchestrator_model=powerful_model,
    default_agent_model="fast",
)

# Execute with natural language
result = swarm.execute("Research the latest AI trends and write a summary report")

print(f"Status: {result.status}")
print(f"Agents spawned: {result.agents_spawned}")
print(f"Tasks completed: {result.tasks_created}")
print(f"Final response: {result.final_response}")
```

## Streaming Output

Use `stream_async()` to capture and print the full execution trajectory (planning + graph events):

```python
import asyncio

async def run():
    swarm = DynamicSwarm(...)
    async for event in swarm.stream_async("Research AI trends and write a summary report"):
        # Print or persist events to build a custom trajectory view
        ...

asyncio.run(run())
```

<details>
<summary>Example output (see examples/dynamic_swarm.py)</summary>

```
============================================================
🚀 DYNAMIC SWARM STARTING
============================================================

📝 Query: Research the latest AI trends and write a summary report
📦 Available tools: ['search_web', 'analyze_data', 'write_file', 'execute_code']
🧠 Available models: ['powerful', 'fast']

----------------------------------------
📐 PHASE 1: PLANNING
----------------------------------------

········································
✅ PLAN READY
········································

Agents (2):
  - researcher: Researches the latest AI trends
  - report_writer: Writes a summary report on the research findings

Tasks (2):
  - research_ai_trends -> researcher
  - write_summary_report -> report_writer (depends: ['research_ai_trends'])

----------------------------------------
⚡ PHASE 2: EXECUTION
----------------------------------------

----------------------------------------
🏁 EXECUTION COMPLETE
----------------------------------------
Status: completed

============================================================
✅ SWARM COMPLETED SUCCESSFULLY
============================================================
```

</details>

## Status & Roadmap

> **Current version: Rollout-only**
>
> This release supports **rollout execution** (string-in, string-out) — ideal for inference and deployment.
>
> **Coming soon:** RL support via [strands-sglang](https://github.com/strands-agents/strands-sglang) integration.

- [x] Rollout execution — string-in, string-out multi-agent workflows
- [x] Dynamic orchestration — automatic agent creation and task assignment
- [x] Streaming trajectory output — consume `stream_async()`
- [ ] RL support — training and fine-tuning via strands-sglang

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## License

Apache-2.0
