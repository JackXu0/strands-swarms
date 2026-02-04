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
    return "[Analysis]\nKey insights: ..."

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
print(f"Agents spawned: {result.agents_spawned_count}")
for agent in result.agents_spawned:
    print(f"  - {agent.name}: {agent.role}")
print(f"Tasks created: {result.tasks_created_count}")
for task in result.tasks_created:
    depends = f" (depends: [{', '.join(task.depends_on)}])" if task.depends_on else ""
    print(f"  - {task.name} -> {task.agent}{depends}")
print(f"Final response: {result.final_response}")
