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

## Why DynamicSwarm?

| Traditional Multi-Agent | DynamicSwarm |
|------------------------|--------------|
| Manually define agents and roles | Agents spawned based on task needs |
| Explicitly wire dependencies | Dependencies inferred from query |
| Configure graph/swarm structure | Structure generated automatically |
| Update code when workflow changes | Same code, different queries |

**Key capabilities:**
- **Automatic workflow planning** — LLM analyzes your query and designs the execution plan
- **Dynamic agent spawning** — Creates only the agents needed, with appropriate tools
- **Dependency-aware execution** — Tasks run in parallel when possible, sequential when required
- **Zero configuration** — No graphs to define, no handoffs to code

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

## Verbose Output

Enable `verbose=True` to see real-time execution with color-coded agents:

```python
swarm = DynamicSwarm(..., verbose=True)
```

<details>
<summary>Example output (colors: 🔵 researcher, 🟢 report_writer)</summary>

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
<thinking>
To analyze this request and design a workflow, we need:

1. A researcher agent to gather information on the latest AI trends using the search_web tool
2. A writer agent to create the summary report using the write_file tool

The research should be completed before the writing can begin, so there is a dependency between the tasks. 

The search_web and write_file tools provide the key capabilities needed. The analyze_data and execute_code tools are not directly relevant for this request.

The "powerful" model should be used for the agents to ensure high-quality research and writing. The "fast" model is less critical for this offline task.

All the necessary tools and information are available to proceed with creating the agents and tasks to fulfill this request.
</thinking>
Tool #1: spawn_agent
Tool #2: spawn_agent
Tool #3: create_task
Tool #4: create_task
Tool #5: finalize_plan

········································
🤖 AGENTS
········································

  🔵 [researcher]
       Role: Researches the latest AI trends
       Tools: ['search_web']
       Model: powerful

  🟢 [report_writer]
       Role: Writes a summary report on the research findings
       Tools: ['write_file']
       Model: powerful

········································
📋 TASKS & DEPENDENCIES
········································

  🔵 [research_ai_trends] → researcher
       Research the latest trends and advancements in artificial intelligence
       ⚡ Can start immediately

  🟢 [write_summary_report] → report_writer
       Write a summary report on the AI trends research findings
       ⏳ Waits for: 🔵 research_ai_trends

········································
✅ PLAN READY — 2 agents, 2 tasks
········································

----------------------------------------
⚡ PHASE 2: EXECUTION
----------------------------------------

🔵 ▶️  research_ai_trends
   <thinking>
   To research the latest AI trends, the most relevant tool is search_web. 
   This will allow me to find up-to-date information on AI trends.
   </thinking>

   Tool: search_web
     query: "latest AI trends"

   <result>
   Summary Report: Latest AI Trends

   Trend 1: Large Language Models — Models like GPT-3, PaLM, and Chinchilla 
   have demonstrated remarkable language understanding and generation...

   Trend 2: Multimodal AI — DALL-E, Imagen, Stable Diffusion can generate 
   highly realistic images from text descriptions...

   Trend 3: Robotics and Embodied AI — Advances in robotic perception, 
   motor control, and task learning for real-world environments...
   </result>
   ✓ Completed

🟢 ▶️  write_summary_report  
   <thinking>
   The research findings are ready. I'll structure the key trends into a 
   clear summary report covering LLMs, multimodal AI, and robotics.
   </thinking>

   <result>
   Summary Report: Latest AI Trends
   [Full report content...]
   </result>
   ✓ Completed

----------------------------------------
🏁 EXECUTION COMPLETE
----------------------------------------
   Status: COMPLETED
   Agents: 2 | Tasks: 2

<thinking>
The agents have completed their tasks. The report covers:
1. Large language models becoming increasingly powerful
2. Multimodal AI advancing rapidly  
3. Progress in robotics and embodied AI
</thinking>

<result>
Here is a summary report on the latest trends in artificial intelligence:

AI continues to advance rapidly across several key fronts. Large language models 
like GPT-3 and PaLM exhibit increasingly powerful natural language abilities. 
Multimodal AI can now generate realistic images from text descriptions. Robots 
are becoming more autonomous with advances in perception and motor control.

The rapid pace of progress looks set to continue and accelerate in coming years.
</result>

============================================================
✅ SWARM COMPLETED SUCCESSFULLY
============================================================
```

</details>

## Custom Event Handling

Use strands-compatible hooks for programmatic event handling:

```python
from strands_swarms import (
    DynamicSwarm,
    HookProvider,
    HookRegistry,
    AgentSpawnedEvent,
    TaskCompletedEvent,
    SwarmCompletedEvent,
)

class MyHookProvider(HookProvider):
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AgentSpawnedEvent, self._on_agent)
        registry.add_callback(TaskCompletedEvent, self._on_task)
        registry.add_callback(SwarmCompletedEvent, self._on_complete)
    
    def _on_agent(self, event: AgentSpawnedEvent) -> None:
        print(f"🤖 Agent '{event.name}' spawned")
    
    def _on_task(self, event: TaskCompletedEvent) -> None:
        print(f"✓ Task '{event.name}' completed")
    
    def _on_complete(self, event: SwarmCompletedEvent) -> None:
        print(f"🏁 Done! {event.agents_spawned} agents, {event.tasks_completed} tasks")

swarm = DynamicSwarm(..., hooks=[MyHookProvider()])
```

Available events: `AgentSpawnedEvent`, `TaskCreatedEvent`, `TaskStartedEvent`, `TaskCompletedEvent`, `SwarmCompletedEvent`, `SwarmFailedEvent`

## When to Use the SDK Directly

`DynamicSwarm` is for **dynamic workflows** where the structure is determined at runtime from natural language.

For **static workflows** where you know the agents and structure upfront, use the official Strands SDK directly:

```python
from strands import Agent
from strands.multiagent.swarm import Swarm
from strands.multiagent.graph import GraphBuilder

# Swarm - dynamic handoffs with shared context
researcher = Agent(name="researcher", system_prompt="You research topics.")
analyzer = Agent(name="analyzer", system_prompt="You analyze data.")
swarm = Swarm([researcher, analyzer])
result = swarm("Research and analyze AI trends")

# Graph - explicit dependency-based execution
builder = GraphBuilder()
builder.add_node(researcher, "research")
builder.add_node(analyzer, "analyze")
builder.add_edge("research", "analyze")
graph = builder.build()
result = graph("Research AI trends")
```

See the SDK docs:
- [`strands.multiagent.swarm.Swarm`](https://github.com/strands-agents/sdk-python/blob/main/src/strands/multiagent/swarm.py)
- [`strands.multiagent.graph.Graph`](https://github.com/strands-agents/sdk-python/blob/main/src/strands/multiagent/graph.py)

## Inspiration

This project is inspired by [Kimi K2.5's Agent Swarm](https://www.kimi.com/blog/kimi-k2-5.html) — where a trainable orchestrator dynamically creates and coordinates sub-agents without predefined roles or workflows. The goal is to build an open-source foundation for training orchestrators that can spin up agent swarms on the fly.

## Status & Roadmap

> **Current version: Rollout-only**
>
> This release supports **rollout execution** (string-in, string-out) — ideal for inference and deployment.
>
> **Coming soon:** RL support via [strands-sglang](https://github.com/strands-agents/strands-sglang) integration.

- [x] Rollout execution — string-in, string-out multi-agent workflows
- [x] Dynamic orchestration — automatic agent creation and task assignment
- [x] Event-driven monitoring — real-time status with hooks
- [ ] RL support — training and fine-tuning via strands-sglang

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## License

Apache-2.0
