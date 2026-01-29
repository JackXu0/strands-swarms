# strands-swarms

Dynamic multi-agent orchestration for [Strands Agents](https://github.com/strands-agents/sdk-python).

Given a query, the swarm automatically plans the workflow, spawns specialized sub-agents, and executes tasks with dependencies.

## What This Package Does

**DynamicSwarm** is the unique value here: an LLM-driven orchestrator that automatically designs and executes multi-agent workflows from natural language queries.

The orchestrator handles three main responsibilities:
1. **Planning and Creating Subagents** - Analyze the task and spawn specialized agents
2. **Assigning Tasks** - Create and assign tasks to the spawned agents with dependencies
3. **Generating Final Response** - Synthesize results into a cohesive response

For **static multi-agent workflows**, use the official Strands SDK directly:
- [`strands.multiagent.swarm.Swarm`](https://github.com/strands-agents/sdk-python/blob/main/src/strands/multiagent/swarm.py) - dynamic handoffs with shared context
- [`strands.multiagent.graph.Graph`](https://github.com/strands-agents/sdk-python/blob/main/src/strands/multiagent/graph.py) - dependency-based execution

## Status

> **Current version: Rollout-only**
>
> This release supports **rollout execution** (string-in, string-out) - ideal for inference and deployment scenarios where you need multi-agent workflows.
>
> **Coming soon:** RL (Reinforcement Learning) support via integration with [strands-sglang](https://github.com/strands-agents/strands-sglang).

## Installation

```bash
pip install git+https://github.com/JackXu0/strands-swarms.git
```

## Quick Start

```python
from strands import tool
from strands.models import BedrockModel
from strands_swarms import DynamicSwarm

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

powerful_model = BedrockModel(model_id="us.anthropic.claude-3-opus-20240229-v1:0")
fast_model = BedrockModel(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0")

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
    verbose=True,
)

result = swarm.execute("Research the latest AI trends and write a summary report")

print(f"Status: {result.status}")
print(f"Agents spawned: {result.agents_spawned}")
print(f"Tasks created: {result.tasks_created}")
print(f"Final response: {result.final_response}")
```

## How It Works

![Architecture](assets/architecture.png)

```
Query: "Research AI trends and write a summary report"
                         │
                         ▼
         ┌───────────────────────────────┐
         │      PHASE 1: PLANNING        │
         │    Orchestrator analyzes      │
         │    query and designs workflow │
         └───────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
    spawn_agent                   spawn_agent
    "researcher"                  "report_writer"
    tools: [search_web]           tools: [write_file]
          │                             │
          ▼                             ▼
    create_task                   create_task
    "research_ai_trends"          "write_summary_report"
    depends_on: []                depends_on: [research_ai_trends]
                         │
                         ▼
         ┌───────────────────────────────┐
         │      PHASE 2: EXECUTION       │
         │    Tasks run based on their   │
         │    dependencies (parallel     │
         │    when possible)             │
         └───────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │   PHASE 3: FINAL RESPONSE     │
         │   Orchestrator synthesizes    │
         │   all task outputs            │
         └───────────────────────────────┘
                         │
                         ▼
                   Final Result
```

## Example Output

With `verbose=True`, you get rich status updates as the swarm executes. Each agent is assigned a unique color via ANSI escape codes for easy tracking.

```
============================================================
🚀 DYNAMIC SWARM STARTING
============================================================

📝 Query: Research the latest AI trends and write a summary report
📦 Available tools: ['search_web', 'analyze_data', 'write_file']
🧠 Available models: ['powerful', 'fast']

----------------------------------------
📐 PHASE 1: PLANNING
----------------------------------------

Tool: spawn_agent
  name: researcher
  role: Researches the latest AI trends

Tool: spawn_agent
  name: report_writer
  role: Writes a summary report on the research findings

Tool: create_task
  name: research_ai_trends
  agent_name: researcher

Tool: create_task
  name: write_summary_report
  agent_name: report_writer
  depends_on: ['research_ai_trends']

Tool: finalize_plan

········································
🤖 AGENTS
········································

  [researcher]                                        # 🔵 Blue
    Role: Researches the latest AI trends
    Tools: ['search_web']
    Model: powerful

  [report_writer]                                     # 🟢 Green
    Role: Writes a summary report on the research findings
    Tools: ['write_file']
    Model: powerful

········································
📋 TASKS & DEPENDENCIES
········································

  [research_ai_trends]
    Agent: researcher
    Description: Research the latest trends and advancements in AI
    ⚡ Can start immediately

  [write_summary_report]
    Agent: report_writer
    Description: Write a summary report on the AI trends research
    ⏳ Waits for: research_ai_trends

········································
✅ PLAN READY
········································
  Entry: auto
  Total: 2 agents, 2 tasks

----------------------------------------
⚡ PHASE 2: EXECUTION
----------------------------------------
📋 Tasks to execute: [research_ai_trends, write_summary_report]

▶️  Executing task: research_ai_trends
   Agent role: Researches the latest AI trends

Tool: search_web
  query: latest AI trends 2024

Summary Report: Latest AI Trends

Artificial intelligence continues to rapidly advance, with several key trends:
- Large Language Models becoming more powerful and capable
- Multimodal AI understanding language, vision, audio together
- Robotics and Embodied AI making progress in real-world tasks

   ✓ Completed: research_ai_trends

▶️  Executing task: write_summary_report
   Agent role: Writes a summary report on the research findings

Tool: write_file
  path: ai_trends_report.md
  content: # Latest AI Trends Report...

   ✓ Completed: write_summary_report

----------------------------------------
🏁 EXECUTION COMPLETE
----------------------------------------
   Status: Status.COMPLETED
   Agents used: 2
   Tasks completed: 2

============================================================
✅ SWARM COMPLETED SUCCESSFULLY
============================================================
```

## Execution Modes

The orchestrator uses dependency-based execution:

### Graph Mode (Default)
Uses the SDK's `Graph` - tasks run based on dependency order with automatic output propagation. Tasks without dependencies run in parallel, while dependent tasks wait for their prerequisites to complete.

## Custom Event Handling

Use strands-compatible hooks for custom event handling:

```python
import time
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

# Use custom hooks instead of verbose=True
swarm = DynamicSwarm(
    available_tools={...},
    available_models={...},
    hooks=[TimestampedHookProvider()],
)
```

## Using the SDK Directly

For static workflows where you know the agents and structure upfront, use the Strands SDK directly:

```python
from strands import Agent
from strands.multiagent.swarm import Swarm
from strands.multiagent.graph import GraphBuilder

# Swarm - dynamic handoffs
researcher = Agent(name="researcher", system_prompt="You research topics.")
analyzer = Agent(name="analyzer", system_prompt="You analyze data.")
swarm = Swarm([researcher, analyzer])
result = swarm("Research and analyze AI trends")

# Graph - dependency-based
builder = GraphBuilder()
builder.add_node(researcher, "research")
builder.add_node(analyzer, "analyze")
builder.add_edge("research", "analyze")
graph = builder.build()
result = graph("Research AI trends")
```

## Roadmap

- [x] **Rollout execution** - String-in, string-out multi-agent workflows
- [x] **Dynamic orchestration** - Orchestrator creates sub-agents, assigns tasks, and generates final response
- [x] **Event-driven monitoring** - Real-time status with hooks
- [ ] **Human-in-the-loop** - Interrupt support for human review during execution
- [ ] **RL support** - Training and fine-tuning via strands-sglang integration

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

Apache-2.0
