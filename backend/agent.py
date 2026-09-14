"""
PlacementPrep AI — Milestone 1: Conversational Interview Agent

This script implements a minimal, beginner-friendly conversational interview agent using:
- LangChain Core Messages (HumanMessage, AIMessage, SystemMessage)
- LangGraph StateGraph & MessagesState
- Google Gemini via langchain-google-genai (Model: gemini-3.5-flash)
- LangGraph InMemorySaver for multi-turn conversation memory with thread_id
- python-dotenv for secure environment variable management
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Silence verbose Google GenAI internal SDK logs for clean CLI experience
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)

# Ensure the project root is in sys.path so modules can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from backend.memory import create_memory_saver, get_session_config
from backend.prompts import get_interviewer_prompt

# Exact Gemini model used for this milestone
GEMINI_MODEL = "gemini-3.5-flash"


def extract_message_text(content: object) -> str:
    """
    Extract clean string text from message content.
    Handles both standard str and list-of-parts (dicts/strings) from Gemini responses.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def create_interviewer_llm(
    model_name: str = GEMINI_MODEL,
    api_key: Optional[str] = None,
) -> ChatGoogleGenerativeAI:
    """
    Initialize and return the Google Gemini Chat model.

    Reads GEMINI_API_KEY from the environment or .env file.
    Uses GEMINI_API_KEY consistently and removes any dependency on GOOGLE_API_KEY.
    Raises a clear ValueError if no API key is found.
    """
    if api_key is None:
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key.strip() == "" or "your_gemini_api_key_here" in api_key:
        raise ValueError(
            "\n[ERROR] GEMINI_API_KEY not found!\n"
            "Please create a .env file in the project root with:\n"
            "GEMINI_API_KEY=your_actual_gemini_api_key\n"
            "Get your key at: https://aistudio.google.com/"
        )

    # Ensure GOOGLE_API_KEY is unset in the process so langchain-google-genai
    # does not emit duplicate key warnings.
    os.environ.pop("GOOGLE_API_KEY", None)

    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.7,
    )


def build_interview_agent(
    subject: str = "Python",
    llm: Optional[object] = None,
    checkpointer: Optional[InMemorySaver] = None,
):
    """
    Build and compile the LangGraph interview agent.

    Args:
        subject: The interview technical topic (default: "Python").
        llm: A LangChain-compatible chat model. If None, initializes Gemini 1.5 Flash.
        checkpointer: LangGraph checkpointer for memory persistence. If None, creates InMemorySaver.

    Returns:
        A compiled LangGraph application with memory checkpointer attached.
    """
    if llm is None:
        llm = create_interviewer_llm()

    if checkpointer is None:
        checkpointer = create_memory_saver()

    system_prompt_text = get_interviewer_prompt(subject=subject)

    # Node: Interacts with the LLM using conversation history and system instructions
    def interviewer_node(state: MessagesState) -> dict:
        # Prepend the system prompt so the LLM remembers its persona and rules on every call
        conversation = [SystemMessage(content=system_prompt_text)] + list(state["messages"])
        ai_response = llm.invoke(conversation)
        # LangGraph's MessagesState automatically appends this return value to state["messages"]
        return {"messages": [ai_response]}

    # Define StateGraph with MessagesState
    workflow = StateGraph(MessagesState)

    # Add the single interviewer node
    workflow.add_node("interviewer", interviewer_node)

    # Define edges: START -> interviewer -> END
    workflow.add_edge(START, "interviewer")
    workflow.add_edge("interviewer", END)

    # Compile with InMemorySaver to persist session state
    agent_app = workflow.compile(checkpointer=checkpointer)
    return agent_app


def run_cli_interview(subject: str = "Python", thread_id: str = "session-1"):
    """
    Run an interactive command-line mock interview session.

    Args:
        subject: Target technical topic.
        thread_id: Unique session ID used by InMemorySaver to track conversation history.
    """
    print("\n" + "=" * 65)
    print("  PlacementPrep AI -- Agentic Mock Interview Assistant")
    print(f"  Subject : {subject}")
    print(f"  Model   : {GEMINI_MODEL}")
    print(f"  Session : {thread_id}")
    print("=" * 65)
    print("Type 'exit' or 'quit' at any time to end the interview.\n")

    # Step 1: Initialize agent
    try:
        app = build_interview_agent(subject=subject)
    except Exception as e:
        print(f"{e}")
        return

    # Session configuration containing the thread_id
    config = get_session_config(thread_id)

    # Step 2: Trigger the first question from the interviewer
    print("[System] Connecting to PlacementPrep AI interviewer...\n")
    try:
        initial_input = {
            "messages": [
                HumanMessage(content=f"Hello! I am ready to begin my mock interview on {subject}.")
            ]
        }
        result = app.invoke(initial_input, config=config)
        first_ai_message = extract_message_text(result["messages"][-1].content)
        print(f"Interviewer: {first_ai_message}\n")
    except Exception as err:
        print(f"\n[ERROR] Failed to communicate with Gemini model: {err}\n")
        return

    # Step 3: Interactive multi-turn loop
    turn = 1
    while True:
        try:
            candidate_answer = input("Candidate > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated by user.")
            break

        if not candidate_answer:
            print("Please enter an answer, or type 'exit' to quit.\n")
            continue

        if candidate_answer.lower() in ["exit", "quit"]:
            print("\nInterviewer: Thank you for practicing with PlacementPrep AI! Best of luck with your placements!")
            print("Session ended.\n")
            break

        turn += 1
        print(f"\n[Turn {turn}] Evaluating answer and generating follow-up...\n")

        try:
            user_message = {"messages": [HumanMessage(content=candidate_answer)]}
            result = app.invoke(user_message, config=config)
            ai_response = extract_message_text(result["messages"][-1].content)
            print(f"Interviewer: {ai_response}\n")
        except Exception as err:
            print(f"\n[ERROR] Model call failed on turn {turn}: {err}\n")
            break


if __name__ == "__main__":
    run_cli_interview(subject="Python", thread_id="interview-session-1")
