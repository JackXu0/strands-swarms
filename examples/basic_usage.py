"""Basic usage examples for strands-swarms."""

from strands import Agent
from strands_swarms import Task, TaskGraph, TaskSwarm, TaskContext, task


# =============================================================================
# Example 1: TaskGraph - Deterministic Pipeline
# =============================================================================

def example_task_graph() -> None:
    """Build a deterministic task pipeline."""
    # Define agents
    researcher = Agent(name="researcher", system_prompt="You research topics concisely.")
    analyzer = Agent(name="analyzer", system_prompt="You analyze data and extract insights.")
    writer = Agent(name="writer", system_prompt="You write clear summaries.")

    # Define tasks with dependencies
    research = Task("research", executor=researcher, description="Research AI trends")
    analyze = Task("analyze", executor=analyzer).depends_on(research)
    summarize = Task("summarize", executor=writer).depends_on(analyze)

    # Build graph
    graph = TaskGraph([research, analyze, summarize])

    print("=== TaskGraph Example ===")
    print(f"Tasks: {list(graph.tasks.keys())}")
    print("Execution order: research -> analyze -> summarize")

    # Uncomment to execute:
    # result = graph.execute("Research AI agent market trends")
    # print(f"Status: {result.status}")


# =============================================================================
# Example 2: TaskSwarm - Dynamic Collaboration
# =============================================================================

def example_task_swarm() -> None:
    """Build a dynamic task swarm."""
    researcher = Agent(name="researcher", system_prompt="You research. Hand off to analyst when done.")
    analyst = Agent(name="analyst", system_prompt="You analyze. Hand off to writer when done.")
    writer = Agent(name="writer", system_prompt="You write final reports.")

    research = Task("research", executor=researcher)
    analyze = Task("analyze", executor=analyst)
    write = Task("write", executor=writer)

    # Swarm allows dynamic handoffs between tasks
    swarm = TaskSwarm([research, analyze, write], entry_task="research")

    print("\n=== TaskSwarm Example ===")
    print(f"Tasks: {list(swarm.tasks.keys())}")
    print("Execution: dynamic handoffs based on agent decisions")

    # Uncomment to execute:
    # result = swarm.execute("Research AI trends and write a report")
    # print(f"Task history: {result.task_history}")


# =============================================================================
# Example 3: Same Agent, Multiple Tasks
# =============================================================================

def example_reusable_agent() -> None:
    """One agent handling multiple tasks."""
    llm = Agent(name="llm", system_prompt="You are a versatile assistant.")

    # Same agent, different tasks
    translate = Task("translate", executor=llm, description="Translate to French")
    summarize = Task("summarize", executor=llm).depends_on(translate)
    format_task = Task("format", executor=llm).depends_on(summarize)

    graph = TaskGraph([translate, summarize, format_task])

    print("\n=== Reusable Agent Example ===")
    for name, task in graph.tasks.items():
        print(f"  {name}: executor={task.executor.name}")


# =============================================================================
# Example 4: Decorator Pattern
# =============================================================================

researcher_agent = Agent(name="researcher", system_prompt="Research specialist")
analyzer_agent = Agent(name="analyzer", system_prompt="Data analyst")


@task(executor=researcher_agent)
def research_trends(ctx: TaskContext) -> str:
    """Research market trends."""
    return f"Researching: {ctx.original_input}"


@task(executor=analyzer_agent, depends_on=[research_trends])
def analyze_trends(ctx: TaskContext) -> str:
    """Analyze the research."""
    return f"Analyzing: {ctx.original_input}"


def example_decorator_pattern() -> None:
    """Build graph from decorated functions."""
    graph = TaskGraph.from_decorated([research_trends, analyze_trends])

    print("\n=== Decorator Pattern Example ===")
    for name, task in graph.tasks.items():
        deps = [d.name for d in task.dependencies]
        print(f"  {name} -> {deps or 'entry point'}")


# =============================================================================
# Example 5: Parallel Tasks (Diamond Pattern)
# =============================================================================

def example_parallel_tasks() -> None:
    """Diamond pattern with parallel execution."""
    agent = Agent(name="worker", system_prompt="General worker")

    fetch = Task("fetch", executor=agent)
    process_a = Task("process_a", executor=agent).depends_on(fetch)
    process_b = Task("process_b", executor=agent).depends_on(fetch)
    merge = Task("merge", executor=agent).depends_on(process_a, process_b)

    graph = TaskGraph([fetch, process_a, process_b, merge])

    print("\n=== Parallel Tasks (Diamond) Example ===")
    print("  fetch")
    print("   ├── process_a ──┐")
    print("   └── process_b ──┴── merge")


if __name__ == "__main__":
    example_task_graph()
    example_task_swarm()
    example_reusable_agent()
    example_decorator_pattern()
    example_parallel_tasks()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Uncomment execute() calls to run with real agents.")
