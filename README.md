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

# 1. Define tools that spawned agents can use
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # In a real implementation, this would call a search API
    return f"[Search Results for '{query}']\n- Result 1: Latest developments..."

@tool
def analyze_data(data: str) -> str:
    """Analyze data and extract insights."""
    return f"[Analysis]\nKey insights: ..."

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    # In a real implementation, this would write to disk
    return f"Successfully wrote {len(content)} characters to {path}"

@tool
def execute_code(code: str) -> str:
    """Execute Python code safely."""
    # In a real implementation, this would use a sandbox
    return f"[Code Output]\nExecuted successfully."

# 2. Create Model instances from strands
# You can use any strands Model: BedrockModel, AnthropicModel, LiteLLMModel, etc.
powerful_model = BedrockModel(model_id="us.anthropic.claude-3-opus-20240229-v1:0")
fast_model = BedrockModel(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0")

# 3. Create the swarm
swarm = DynamicSwarm(
    # Tools that spawned agents can use
    available_tools={
        "search_web": search_web,
        "analyze_data": analyze_data,
        "write_file": write_file,
        "execute_code": execute_code,
    },
    # Models that spawned agents can use
    # Keys are friendly names the orchestrator uses, values are Model instances
    available_models={
        "powerful": powerful_model,
        "fast": fast_model,
    },
    # Model instance for the orchestrator agent
    orchestrator_model=powerful_model,
    # Default model name for spawned agents if not specified
    default_agent_model="fast",
    
    # Event handling options:
    # Option 1: verbose=True uses PrintingHookProvider (rich CLI output)
    verbose=True,
    # Option 2: Custom hook provider for your own logging/UI
    # hooks=[MyCustomHookProvider()],
)

# 4. Execute - the orchestrator will:
#    1. Analyze the query
#    2. Decide what agents to spawn (researcher, writer, etc.)
#    3. Assign tools and models to each agent
#    4. Create tasks with proper dependencies
#    5. Execute the workflow
result = swarm.execute("Research the latest AI trends and write a summary report")

# 5. Inspect results
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
    │    ORCHESTRATOR PHASE 1 & 2   │
    │  1. Plan & Create Subagents   │
    │  2. Assign Tasks              │
    └───────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   spawn_agent              spawn_agent
   "researcher"             "writer"
   tools: [search_web]      tools: [write_file]
        │                       │
        ▼                       ▼
   create_task              create_task
   "research"               "write"
   depends_on: []           depends_on: [research]
                    │
                    ▼
    ┌───────────────────────────────┐
    │          EXECUTION            │
    │   Tasks run based on their    │
    │   dependencies (Graph mode)   │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │     ORCHESTRATOR PHASE 3      │
    │  3. Generate Final Response   │
    │  Synthesize all task outputs  │
    └───────────────────────────────┘
                    │
                    ▼
              Final Result
```

## Example Output

With `verbose=True`, you get rich status updates as the swarm executes. Each agent is assigned a unique color via ANSI escape codes for easy tracking.

```
Query: Research the latest AI trends and write a summary report

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
🤖 AGENTS                                            # Colors: 🔵 Blue, 🟢 Green
········································

  [researcher]
    Role: Researches the latest AI trends
    Tools: ['search_web']
    Model: powerful

  [report_writer]
    Role: Writes a summary report on the research findings
    Tools: ['write_file']
    Model: powerful

········································
📋 TASKS & DEPENDENCIES
········································

  [research_ai_trends]
    Agent: researcher
    Description: Research the latest trends and advancements in artificial intelligence
    ⚡ Can start immediately

  [write_summary_report]
    Agent: report_writer
    Description: Write a summary report on the AI trends research findings
    ⏳ Waits for: research_ai_trends

········································
✅ PLAN READY
········································
  Entry: auto
  Total: 2 agents, 2 tasks


<task_quality_reflection>
The workflow looks well designed to handle the request of researching the latest AI trends and writing a summary report:

- It creates two specialized agents, a researcher to gather the AI trend information using search_web, and a report writer to summarize the findings using write_file. This division of labor makes sense.

- The model selected for both agents is "powerful", which is appropriate to ensure high quality research and writing. Speed is less critical for this offline task.

- The dependency is correctly defined, with the writing task waiting on the research task to complete first. This ensures the report will incorporate the research findings.

- All the key steps are covered, from information gathering to report generation. No obvious gaps in the workflow.

So in summary, the agent design, task split, tool usage and ordering all look on target to efficiently produce a quality report on the latest AI trends. The plan seems ready to execute.
</task_quality_reflection>

<task_quality_score>5</task_quality_score>

<result>
The workflow has been designed to research the latest AI trends and write a summary report:

Two specialized agents were created:
- A researcher agent to gather information on AI trends using the search_web tool
- A report writer agent to generate the summary report based on the research, using the write_file tool

The research task was assigned to the researcher agent, and the report writing task to the report writer. A dependency was defined so that report writing will begin only after the research is complete.

Both agents were given the "powerful" model to ensure high quality results. The plan has been finalized and is ready to execute.

Please stand by while the agents complete their assigned tasks to deliver the AI trend report you requested. Let me know if you need anything else!
</result>
----------------------------------------
⚡ PHASE 2: EXECUTION
----------------------------------------
📋 Tasks to execute: [research_ai_trends, write_summary_report]

▶️  Executing task: research_ai_trends
<thinking>
To research the latest AI trends and write a summary report, the most relevant tool is the search_web function. This will allow me to search for the latest information on AI trends.

Required parameters:
- query (string): The query for this would be something like "latest AI trends". This can be inferred from the user's request.

No other tools are needed, as the search results should provide enough information to write a summary report on the latest AI trends.

Since I have the required parameter to call the search_web tool, I will proceed with making that API call.
</thinking>
Tool: search_web


<search_quality_reflection>
The search results provide a good overview of some of the latest AI trends and developments. There are specific examples of new AI capabilities, as well as analysis of the overall direction the field is moving in. This should provide sufficient information to write a high-level summary report on the topic. No additional searches are needed.
</search_quality_reflection>
<search_quality_score>4</search_quality_score>

<result>
Summary Report: Latest AI Trends

Artificial intelligence continues to rapidly advance, with several key trends emerging in the field:

Trend 1: Increasing Power and Capabilities of Large Language Models 
Some of the most impressive recent AI achievements have come from massive neural networks trained on huge text datasets, known as large language models...

Trend 2: Multimodal AI 
Another major trend is AI that can understand and work with multiple modalities, like language, vision, audio, etc...

Trend 3: Robotics and Embodied AI
While a lot of focus has been on disembodied AI that exists in the digital realm, there is also increasing interest and progress in physically embodied AI systems like robots...

In summary, AI capabilities are rapidly progressing on multiple fronts - from language to multimodal understanding to real-world embodiment in robots.
</result>   ✓ Completed: research_ai_trends

▶️  Executing task: write_summary_report
<thinking>
The summary report on the latest AI trends has already been written based on the research findings in the previous steps. No additional tools are needed at this point...
</thinking>

<result>
Summary Report: Latest AI Trends
...
</result>   ✓ Completed: write_summary_report

----------------------------------------
🏁 EXECUTION COMPLETE
----------------------------------------
   Status: Status.COMPLETED
   Agents used: 2
   Tasks completed: 2

============================================================
✅ SWARM COMPLETED SUCCESSFULLY
============================================================

============================================================
RESULTS
============================================================
Status: Status.COMPLETED
Agents spawned: 2
Tasks created: 2

Final response:
<thinking>
The agents have completed their tasks of researching the latest AI trends and writing a comprehensive summary report...
</thinking>

<result>
Here is a summary report on the latest trends in artificial intelligence:

AI continues to advance rapidly across several key fronts. Large language models like GPT-3 and PaLM are exhibiting increasingly powerful natural language abilities, able to engage in open-ended dialogue, answer questions, and generate human-like text. At the same time, multimodal AI that can understand and generate content across modalities like language, images, and audio is also progressing quickly. Models like DALL-E can now produce strikingly realistic images from textual descriptions.

On the embodied AI front, robots are becoming more autonomous and capable, with advances in perception, motor control and task learning. Robots are becoming better at operating in unstructured environments and adapting to novel situations.

Across the board, AI systems are becoming more sophisticated, with growing power and flexibility. Organizations in all industries need to pay attention to these AI advances in order to capitalize on the opportunities and disruptive potential of the technology. The rapid pace of progress in AI looks set to continue and even accelerate in the coming years.
</result>
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
