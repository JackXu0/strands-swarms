# strands-swarms

Dynamic multi-agent orchestration for [Strands Agents](https://github.com/strands-agents/sdk-python).

Given a query, the swarm automatically plans the workflow, spawns specialized sub-agents, and executes tasks with dependencies.

## Status

> **Current version: Rollout-only**
>
> This release supports **rollout execution** (string-in, string-out) - ideal for inference and deployment scenarios where you need multi-agent workflows.
>
> **Coming soon:** RL (Reinforcement Learning) support via integration with [strands-sglang](https://github.com/strands-agents/strands-sglang), enabling token in token out (TITO) to multi agent orchestration model.

## Installation

```bash
pip install git+https://github.com/JackXu0/strands-swarms.git
```

## Quick Start

```python
from strands import tool
from strands.models import BedrockModel
from strands_swarms import DynamicSwarm

# 1. Define tools that sub-agents can use
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def analyze_data(data: str) -> str:
    """Analyze data and extract insights."""
    return f"Analysis: {data}"

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return f"Wrote to {path}"

@tool
def execute_code(code: str) -> str:
    """Execute Python code safely."""
    return f"Executed successfully"

# 2. Define models that sub-agents can use
powerful_model = BedrockModel(model_id="us.anthropic.claude-3-opus-20240229-v1:0")
fast_model = BedrockModel(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0")

# 3. Create the swarm
swarm = DynamicSwarm(
    available_tools={
        "search_web": search_web,
        "analyze_data": analyze_data,
        "write_file": write_file,
        "execute_code": execute_code,
    },
    available_models={
        "powerful": powerful_model,
        "fast": fast_model,
    },
    planner_model=powerful_model,
    default_agent_model="fast",
    verbose=True,  # See live status
)

# 4. Execute - the swarm handles everything
result = swarm.execute("Research the latest AI trends and write a summary report")
```

## How It Works

```
Query: "Research AI trends and write a summary report"
                    │
                    ▼
    ┌───────────────────────────────┐
    │           PLANNING            │
    │   Planner analyzes query and  │
    │   designs the workflow        │
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
    │   Tasks run in dependency     │
    │   order with their agents     │
    └───────────────────────────────┘
                    │
                    ▼
              Final Result
```

## Example Output

With `verbose=True`, you get rich status updates as the swarm executes. Each agent is assigned a unique color via ANSI escape codes to easily track which agent is producing output during execution.

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
To address this request to research AI trends and write a summary report, we will need agents to:
1) Perform web research on the latest AI trends
2) Analyze the research data 
3) Write up the findings in a summary report

For the research agent, the search_web tool will be needed. All required parameters for search_web can be inferred from the request.

To analyze the research, the analyze_data tool will be helpful. It looks like all required parameters can be obtained from the output of the search_web tool.

For writing the report, the write_file tool will be used. The contents for the file can come from the analyze_data output. A filename will need to be specified.

In terms of dependencies, the data analysis should wait until the web research is complete. And the report writing should wait until the data analysis is done.

The "powerful" model should be sufficient for all agents.
</thinking>
Tool #1: spawn_agent

Tool #2: spawn_agent

Tool #3: spawn_agent

Tool #4: create_task

Tool #5: create_task

Tool #6: create_task

Tool #7: execute_swarm

········································
🤖 AGENTS
········································

  🔵 [researcher]
    Role: Perform web research on the latest AI trends
    Tools: ['search_web']
    Model: powerful

  🟢 [analyst]
    Role: Analyze the AI trends research data
    Tools: ['analyze_data']
    Model: powerful

  🟡 [report_writer]
    Role: Write a summary report on the latest AI trends
    Tools: ['write_file']
    Model: powerful

········································
📋 TASKS & DEPENDENCIES
········································

  🔵 [research_ai_trends]
    Agent: researcher
    Description: Search the web for the latest trends in artificial intelligence
    ⚡ Can start immediately

  🟢 [analyze_research]
    Agent: analyst
    Description: Analyze the data gathered from the web research on AI trends
    ⏳ Waits for: research_ai_trends

  🟡 [write_report]
    Agent: report_writer
    Description: Write a summary report on the latest AI trends based on the research and analysis
    ⏳ Waits for: analyze_research

········································
✅ PLAN READY
········································
  Mode: graph (dependency-based)
  Entry: auto
  Total: 3 agents, 3 tasks


The workflow has been set up with the necessary agents and tasks to research AI trends and write a summary report. The key steps:

1. Created a researcher agent to search the web for AI trends
2. Created an analyst agent to analyze the research data  
3. Created a report writer agent to summarize the findings
4. Set up task dependencies so analysis waits for research, and report writing waits for analysis
5. Executing the swarm in "graph" mode to run the tasks based on the dependencies

Let me know if you would like me to modify the workflow in any way. Otherwise, the swarm is ready to execute to research the latest AI trends and generate the summary report.
----------------------------------------
⚡ PHASE 2: EXECUTION
----------------------------------------

🔄 Mode: graph
📋 Tasks to execute: [🔵 research_ai_trends, 🟢 analyze_research, 🟡 write_report]

▶️  Executing task: 🔵 research_ai_trends

🔵 [researcher] <thinking>
To research the latest trends in AI, I will need to use the search_web tool...
</thinking>[calling search_web] 

Here is a summary of the latest AI trends based on my research:

- Large language models like GPT-3 are becoming more powerful...
- AI is being increasingly applied to scientific research...
- Responsible AI is a growing focus...
- Multimodal AI is an active area of research...
- AI hardware is rapidly advancing...
- Industrial application of AI continues to grow...

In summary, AI capabilities are rapidly advancing while also becoming more 
responsible, efficient, multimodal, and industrialized.
   ✓ Completed: 🔵 research_ai_trends

▶️  Executing task: 🟢 analyze_research

🟢 [analyst] <thinking>
The task is to analyze the research data on AI trends...
</thinking>[calling analyze_data] 

Based on the analysis, the key trends and insights regarding AI are:
1. Significant advancements in large language models...
2. AI is having a transformative impact on scientific research...
3. Growing emphasis on responsible AI systems...
4. Multimodal AI is an emerging frontier...
5. Rapid progress in AI hardware...
6. Adoption of AI in industry continues to expand...
   ✓ Completed: 🟢 analyze_research

▶️  Executing task: 🟡 write_report

🟡 [report_writer] <thinking>
To write a summary report, I have the analysis from the previous step...
</thinking>[calling write_file] 

The full report has been written to summary_report.txt.
   ✓ Completed: 🟡 write_report

----------------------------------------
🏁 EXECUTION COMPLETE
----------------------------------------
   Status: Status.COMPLETED
   Agents used: 3
   Tasks completed: 3
Here is the summary report on the latest AI trends:

Summary Report: Latest AI Trends

Artificial intelligence technology is advancing rapidly across several key dimensions:

- Large language models are becoming much more capable and efficient, able to perform a wider range of natural language tasks with less training data. Research is focused on further improving their controllability and adaptability.

- AI is accelerating scientific discovery by uncovering insights from large datasets, with transformative impact in fields like drug development, materials science, and climate modeling. 

- There is growing emphasis on developing responsible AI systems that are explainable, fair, private, secure, and aligned with human values. This involves both technical approaches and crucial interdisciplinary work across ethics, policy, and social impact.

- Multimodal AI that can understand and generate content across text, images, video and audio is an emerging frontier, with generative models hinting at the multimedia AI capabilities that may soon be possible.

- Rapid progress in specialized AI hardware is providing the immense computational power required to efficiently train and run increasingly sophisticated models, including deploying AI on edge devices.

- Adoption of AI in industry continues to expand across sectors for use cases like automation, analytics, and personalization. MLOps tools and enterprise AI platforms are helping to scale and manage AI in production.

In summary, AI is advancing quickly in raw capabilities while also maturing in important dimensions like responsible development and industrial-grade tooling. It is driving breakthroughs in science and enabling intelligent applications across industries. Guided by work to ensure AI systems remain robust and beneficial, the coming years will likely bring AI that is simultaneously far more powerful, dependable and pervasive.
============================================================
✅ SWARM COMPLETED SUCCESSFULLY
============================================================
```

> **Note:** The colored circles (🔵🟢🟡) represent ANSI color codes in your terminal. Each agent gets a unique color for easy visual tracking during execution.

## Custom Event Handling

Use strands-compatible hooks for custom event handling:

```python
from strands_swarms import (
    DynamicSwarm,
    HookProvider,
    HookRegistry,
    AgentSpawnedEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
)

class MyHooks(HookProvider):
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AgentSpawnedEvent, self.on_agent)
        registry.add_callback(TaskStartedEvent, self.on_task_started)
        registry.add_callback(TaskCompletedEvent, self.on_task_completed)
    
    def on_agent(self, event: AgentSpawnedEvent) -> None:
        print(f"Agent '{event.name}' spawned with role: {event.role}")
    
    def on_task_started(self, event: TaskStartedEvent) -> None:
        print(f"Task '{event.name}' started")
    
    def on_task_completed(self, event: TaskCompletedEvent) -> None:
        print(f"Task '{event.name}' completed")

swarm = DynamicSwarm(..., hooks=[MyHooks()])
```

## Roadmap

- [x] **Rollout execution** - String-in, string-out multi-agent workflows
- [x] **Dynamic agent spawning and task planning** - Planner agent automatically creates specialized sub-agents, tasks, and dependencies from natural language
- [x] **Event-driven execution** - Real-time monitoring with hooks and dynamic inter-task handoffs via `TaskSwarm`
- [ ] **RL support** - Training and fine-tuning via strands-sglang integration
- [ ] **Reward shaping** - Custom reward functions for multi-agent optimization

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

Apache-2.0
