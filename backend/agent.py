"""
PlacementPrep AI — Milestone 1, 2, & 3: Agentic Conversational Interview Assistant

Milestone 1: Conversational engine, StateGraph, MessagesState, InMemorySaver, thread_id session memory.
Milestone 2: CandidateProfile extraction and personalized, role-grounded questioning.
Milestone 3: Agentic RAG with confidence gating, answer evaluation, deterministic difficulty updates, and decision logging.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Silence verbose Google GenAI internal SDK logs for clean CLI experience
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)

# Ensure the project root is in sys.path so modules can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from backend.evaluation import compute_next_difficulty, evaluate_candidate_answer
from backend.memory import create_memory_saver, get_session_config
from backend.models import AnswerEvaluation, CandidateProfile, DecisionLogEntry
from backend.prompts import get_interviewer_prompt
from backend.rag import retrieve_prep_material

# Exact Gemini model used for PlacementPrep AI
GEMINI_MODEL = "gemini-3.5-flash"


class InterviewState(MessagesState):
    """
    State for the PlacementPrep AI interview agent.
    Inherits from MessagesState to preserve all Milestone 1 & 2 message history with add_messages reducer.
    """
    candidate_profile: Optional[CandidateProfile] = None
    subject: str = "Python"
    difficulty: int = 2
    decision_log: List[dict] = []
    evaluation_rating: Optional[str] = None
    evaluation_feedback: Optional[str] = None
    previous_difficulty: Optional[int] = None


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
    candidate_profile: Optional[CandidateProfile] = None,
    llm: Optional[object] = None,
    checkpointer: Optional[InMemorySaver] = None,
    eval_llm: Optional[object] = None,
):
    """
    Build and compile the LangGraph interview agent with Agentic RAG and evaluation.

    Args:
        subject: The interview technical topic (default: "Python").
        candidate_profile: Optional CandidateProfile extracted from candidate resume.
        llm: LangChain-compatible chat model. If None, initializes Gemini 3.5 Flash.
        checkpointer: LangGraph checkpointer for memory persistence. If None, creates InMemorySaver.
        eval_llm: Optional chat model for evaluation node. If None, uses llm.

    Returns:
        A compiled LangGraph application with memory checkpointer attached.
    """
    if llm is None:
        llm = create_interviewer_llm()

    if checkpointer is None:
        checkpointer = create_memory_saver()

    # Bind retrieval tool to interviewer LLM if supported
    if hasattr(llm, "bind_tools"):
        llm_with_tools = llm.bind_tools([retrieve_prep_material])
    else:
        llm_with_tools = llm

    # Node 2: Interviewer generates next response or invokes retrieval tool
    def interviewer_node(state: InterviewState) -> dict:
        current_difficulty = state.get("difficulty", 2)
        active_profile = state.get("candidate_profile") or candidate_profile

        system_prompt = get_interviewer_prompt(
            subject=subject,
            candidate_profile=active_profile,
            difficulty=current_difficulty,
        )

        messages = state["messages"]

        # Loop protection: max 1 retrieval per turn
        last_human_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break

        retrieval_already_called = False
        if last_human_idx is not None:
            for m in messages[last_human_idx:]:
                if isinstance(m, ToolMessage) and m.name == "retrieve_prep_material":
                    retrieval_already_called = True
                    break

        conversation = [SystemMessage(content=system_prompt)] + list(messages)

        # Disable tools if we already retrieved this turn
        if hasattr(llm, "bind_tools") and not retrieval_already_called:
            current_llm = llm_with_tools
        else:
            current_llm = llm

        ai_response = current_llm.invoke(conversation)
        return {"messages": [ai_response]}

    # Routing condition: check if interviewer called a tool
    def route_interviewer(state: InterviewState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "log_decision"

    # Node 1: Evaluate Candidate Answer (Runs at START)
    def evaluate_node(state: InterviewState) -> dict:
        messages = state["messages"]
        current_diff = state.get("difficulty", 2)

        # Find the latest HumanMessage
        last_human_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break

        # Find previous question to evaluate answer against
        prev_question = None
        if last_human_idx is not None:
            for i in range(last_human_idx - 1, -1, -1):
                if isinstance(messages[i], AIMessage) and messages[i].content:
                    prev_question = extract_message_text(messages[i].content)
                    break

        # If this is the initial greeting turn, no technical evaluation needed
        if prev_question is None or last_human_idx is None:
            return {
                "evaluation_rating": "greeting",
                "previous_difficulty": current_diff,
                "difficulty": current_diff,
            }

        # Evaluate candidate's technical answer
        candidate_ans = messages[last_human_idx].content
        feedback_val = None
        try:
            eval_result = evaluate_candidate_answer(
                question=prev_question,
                candidate_answer=candidate_ans,
                subject=subject,
                llm=eval_llm or llm,
            )
            rating = eval_result.rating
            feedback_val = eval_result.feedback
        except Exception:
            rating = "error"

        new_diff = compute_next_difficulty(current_diff, rating)

        return {
            "evaluation_rating": rating,
            "evaluation_feedback": feedback_val,
            "previous_difficulty": current_diff,
            "difficulty": new_diff,
        }

    # Node 3: Log Decision (Runs at END)
    def log_decision_node(state: InterviewState) -> dict:
        messages = state["messages"]
        existing_log = list(state.get("decision_log") or [])
        turn_num = len(existing_log) + 1

        current_diff = state.get("previous_difficulty", 2)
        new_diff = state.get("difficulty", 2)
        rating = state.get("evaluation_rating", "greeting")
        feedback_val = state.get("evaluation_feedback")

        # Find the latest HumanMessage and previous AI question
        last_human_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break

        prev_question = None
        candidate_ans = None
        if last_human_idx is not None:
            candidate_ans = extract_message_text(messages[last_human_idx].content)
            for i in range(last_human_idx - 1, -1, -1):
                if isinstance(messages[i], AIMessage) and messages[i].content:
                    prev_question = extract_message_text(messages[i].content)
                    break

        # Check if retrieval was invoked on this turn
        retrieval_called = False
        retrieval_query = None
        retrieval_confidence = None
        retrieval_used = False
        fallback_used = False

        if last_human_idx is not None:
            for m in messages[last_human_idx:]:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        if tc.get("name") == "retrieve_prep_material":
                            retrieval_called = True
                            retrieval_query = tc.get("args", {}).get("query")
                if isinstance(m, ToolMessage) and m.name == "retrieve_prep_material":
                    try:
                        tool_data = json.loads(m.content)
                        retrieval_confidence = tool_data.get("confidence")
                        status = tool_data.get("status")
                        retrieval_used = (status == "HIGH_CONFIDENCE")
                        fallback_used = (status in ["LOW_CONFIDENCE", "EMPTY"])
                    except Exception:
                        pass

        from backend.rag import CONFIDENCE_THRESHOLD

        score_gap = None
        if retrieval_confidence is not None:
            score_gap = round(retrieval_confidence - CONFIDENCE_THRESHOLD, 3)

        delta = new_diff - current_diff
        delta_str = f"{delta:+d}" if delta != 0 else "0"

        if rating == "greeting":
            if retrieval_called:
                if retrieval_used:
                    decision_reason = (
                        f"Initial turn; retrieved prep material with confidence {retrieval_confidence} "
                        f"(gap {score_gap:+.3f} >= 0.0); prep material applied to initial question."
                    )
                else:
                    decision_reason = (
                        f"Initial turn; retrieval confidence {retrieval_confidence} is below threshold "
                        f"{CONFIDENCE_THRESHOLD} (gap {score_gap:+.3f} < 0.0); fell back to candidate profile and conversation context."
                    )
            else:
                decision_reason = f"Initial interview greeting; baseline difficulty initialized to {current_diff}."
        else:
            if rating == "error":
                eval_note = "Evaluation failed (fallback to error rating)"
            else:
                eval_note = f"Answer rated {rating}"

            if retrieval_called:
                if retrieval_used:
                    decision_reason = (
                        f"{eval_note} (difficulty {current_diff}->{new_diff}, delta {delta_str}). "
                        f"Retrieved prep material with confidence {retrieval_confidence} (gap {score_gap:+.3f} >= 0.0); prep material applied to next question."
                    )
                else:
                    decision_reason = (
                        f"{eval_note} (difficulty {current_diff}->{new_diff}, delta {delta_str}). "
                        f"Retrieval confidence {retrieval_confidence} is below threshold {CONFIDENCE_THRESHOLD} (gap {score_gap:+.3f} < 0.0); fell back to candidate profile and conversation context."
                    )
            else:
                decision_reason = (
                    f"{eval_note} (difficulty {current_diff}->{new_diff}, delta {delta_str}). "
                    f"Retrieval not required; candidate profile and conversation history are sufficient."
                )

        entry = DecisionLogEntry(
            turn=turn_num,
            retrieval_called=retrieval_called,
            retrieval_query=retrieval_query,
            retrieval_confidence=retrieval_confidence,
            retrieval_score_gap=score_gap,
            retrieval_used=retrieval_used,
            fallback_used=fallback_used,
            difficulty_before=current_diff,
            difficulty_after=new_diff,
            difficulty_delta=delta,
            decision_reason=decision_reason,
            evaluation_rating=rating,
            question_text=prev_question,
            candidate_answer=candidate_ans,
            feedback=feedback_val,
        )

        return {
            "decision_log": existing_log + [entry.model_dump()],
        }

    # Construct StateGraph with native LangGraph tool calling
    workflow = StateGraph(InterviewState)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("interviewer", interviewer_node)
    workflow.add_node("tools", ToolNode([retrieve_prep_material]))
    workflow.add_node("log_decision", log_decision_node)

    workflow.add_edge(START, "evaluate")
    workflow.add_edge("evaluate", "interviewer")
    workflow.add_conditional_edges(
        "interviewer",
        route_interviewer,
        ["tools", "log_decision"],
    )
    workflow.add_edge("tools", "interviewer")
    workflow.add_edge("log_decision", END)

    agent_app = workflow.compile(checkpointer=checkpointer)
    return agent_app


def run_cli_interview(
    subject: str = "Python",
    thread_id: str = "session-1",
    resume_path: Optional[str] = None,
    target_role: str = "Junior Python Backend Developer",
):
    """
    Run an interactive command-line mock interview session with Agentic RAG and Decision Logging.

    Args:
        subject: Target technical topic.
        thread_id: Unique session ID used by InMemorySaver to track conversation history.
        resume_path: Optional path to a resume text file for personalized interviews.
        target_role: Target job position when running with resume analysis.
    """
    candidate_profile: Optional[CandidateProfile] = None

    if resume_path:
        resume_file = Path(resume_path)
        if not resume_file.exists():
            print(f"\n[ERROR] Resume file not found at: {resume_path}\n")
            return

        print(f"\n[Milestone 2] Analyzing resume from '{resume_path}' for target role '{target_role}'...")
        try:
            from backend.parser import analyze_resume

            with open(resume_file, "r", encoding="utf-8") as f:
                resume_text = f.read()

            candidate_profile = analyze_resume(resume_text=resume_text, target_role=target_role)

            print("\n" + "=" * 65)
            print("  Candidate Profile Extracted Successfully")
            print("=" * 65)
            print(f"  Name            : {candidate_profile.name}")
            print(f"  Target Role     : {candidate_profile.target_role}")
            print(f"  Skills          : {', '.join(candidate_profile.skills[:8])}...")
            print(f"  Projects ({len(candidate_profile.projects)}):")
            for p in candidate_profile.projects:
                print(f"    - {p.name} [{', '.join(p.technologies)}]")
            print(f"  Interview Focus : {', '.join(candidate_profile.interview_focus)}")
            print(f"  Areas to Probe  : {', '.join(candidate_profile.potential_weak_areas)}")
            print("=" * 65)
        except Exception as e:
            print(f"\n[ERROR] Failed to analyze resume: {e}\n")
            return

    print("\n" + "=" * 65)
    print("  PlacementPrep AI -- Agentic Mock Interview Assistant")
    print(f"  Subject : {subject}")
    print(f"  Model   : {GEMINI_MODEL}")
    print(f"  Session : {thread_id}")
    if candidate_profile:
        print(f"  Candidate: {candidate_profile.name} ({candidate_profile.target_role})")
    print("=" * 65)
    print("Type 'exit' or 'quit' at any time to end the interview.\n")

    # Step 1: Initialize agent
    try:
        app = build_interview_agent(subject=subject, candidate_profile=candidate_profile)
    except Exception as e:
        print(f"{e}")
        return

    config = get_session_config(thread_id)

    # Step 2: Trigger the first question from the interviewer
    print("[System] Connecting to PlacementPrep AI interviewer...\n")
    try:
        if candidate_profile:
            initial_msg = (
                f"Hello! I am {candidate_profile.name}, and I am ready to begin my "
                f"mock technical interview for the role of {candidate_profile.target_role} focusing on {subject}."
            )
        else:
            initial_msg = f"Hello! I am ready to begin my mock interview on {subject}."

        initial_input = {
            "messages": [HumanMessage(content=initial_msg)],
            "difficulty": 2,
            "subject": subject,
            "candidate_profile": candidate_profile,
        }
        result = app.invoke(initial_input, config=config)

        # Get the final AI message
        first_ai_message = ""
        for m in reversed(result["messages"]):
            if isinstance(m, AIMessage) and m.content:
                first_ai_message = extract_message_text(m.content)
                break

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
            name_str = f" {candidate_profile.name}" if candidate_profile else ""
            print(f"\nInterviewer: Thank you{name_str} for practicing with PlacementPrep AI! Best of luck with your placements!")
            print("Session ended.\n")
            break

        turn += 1
        print(f"\n[Turn {turn}] Evaluating answer and determining next step...")

        try:
            user_message = {"messages": [HumanMessage(content=candidate_answer)]}
            result = app.invoke(user_message, config=config)

            ai_response = ""
            for m in reversed(result["messages"]):
                if isinstance(m, AIMessage) and m.content:
                    ai_response = extract_message_text(m.content)
                    break

            # Print Decision Log entry for this turn (for AWS Demo visibility)
            if result.get("decision_log"):
                last_log = result["decision_log"][-1]
                delta_val = last_log.get('difficulty_delta', 0)
                delta_str = f"{delta_val:+d}" if delta_val != 0 else "0"
                print("\n" + "-" * 65)
                print(f"  [Agent Decision Log — Turn {turn}]")
                print(f"  * Answer Evaluation   : {last_log.get('evaluation_rating', 'N/A').upper()}")
                print(f"  * Difficulty Level    : {last_log.get('difficulty_before')} -> {last_log.get('difficulty_after')} (delta: {delta_str})")
                print(f"  * Retrieval Called    : {last_log.get('retrieval_called')}")
                if last_log.get("retrieval_called"):
                    print(f"  * Retrieval Query     : \"{last_log.get('retrieval_query')}\"")
                    print(f"  * Confidence Score    : {last_log.get('retrieval_confidence')}")
                    gap = last_log.get('retrieval_score_gap')
                    gap_str = f"{gap:+.3f}" if gap is not None else "N/A"
                    print(f"  * Score Gap vs Thresh : {gap_str} (threshold: 0.70)")
                    print(f"  * Material Used       : {last_log.get('retrieval_used')}")
                    print(f"  * Fallback Triggered  : {last_log.get('fallback_used')}")
                print(f"  * Decision Reason     : {last_log.get('decision_reason')}")
                print("-" * 65 + "\n")

            print(f"Interviewer: {ai_response}\n")
        except Exception as err:
            print(f"\n[ERROR] Model call failed on turn {turn}: {err}\n")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlacementPrep AI Mock Interview Assistant")
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Path to resume plain text file to enable personalized interview (Milestone 2)",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="Junior Python Backend Developer",
        help="Target job role for the candidate interview (default: 'Junior Python Backend Developer')",
    )
    parser.add_argument(
        "--subject", "-s",
        type=str,
        default="Python",
        help="Technical topic for the interview (default: 'Python')",
    )
    parser.add_argument(
        "--thread", "-t",
        type=str,
        default="interview-session-1",
        help="Session identifier for memory persistence (default: 'interview-session-1')",
    )

    args = parser.parse_args()

    run_cli_interview(
        subject=args.subject,
        thread_id=args.thread,
        resume_path=args.resume,
        target_role=args.role,
    )
