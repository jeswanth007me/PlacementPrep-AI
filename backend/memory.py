"""
Memory management for PlacementPrep AI.

This module sets up LangGraph's InMemorySaver checkpointer.
InMemorySaver preserves conversation state (messages, intermediate values)
in RAM, keyed by a unique session identifier called `thread_id`.
"""

from langgraph.checkpoint.memory import InMemorySaver


def create_memory_saver() -> InMemorySaver:
    """
    Instantiate and return an InMemorySaver checkpointer.
    
    How it works:
    - LangGraph uses checkpointers to save the state of a graph execution.
    - Every time a node runs, the state is persisted under a specific thread_id.
    - Subsequent calls with the same thread_id automatically load previous messages,
      allowing the agent to remember everything discussed earlier in the session.
    """
    return InMemorySaver()


def get_session_config(thread_id: str) -> dict:
    """
    Generate the standard LangGraph configuration dictionary with thread_id.

    Args:
        thread_id: Unique string identifying the interview session (e.g., 'interview-session-1').

    Returns:
        A dictionary with the format: {"configurable": {"thread_id": thread_id}}
    """
    return {"configurable": {"thread_id": thread_id}}
