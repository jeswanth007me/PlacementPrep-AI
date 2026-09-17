"""
PlaceMate AI — Production REST API Layer.

Built on FastAPI + Uvicorn.
Each endpoint delegates to existing backend functions without
modifying the AI core (agent.py, parser.py, evaluation.py,
memory.py, prompts.py, github/, rag.py).

Thread management: backend/session_manager.py
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from backend.memory import get_session_config
from backend.models import CandidateProfile, FinalResult, GitHubProfile
from backend.parser import analyze_resume
from backend.github.analyzer import fetch_and_analyze_github_profile
from backend.results import generate_final_result
from backend.session_manager import session_manager

# Load env
load_dotenv()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PlaceMate AI API",
    description="REST API for the PlaceMate AI mock interview assistant.",
    version="1.0.0",
)

# CORS: allow origins from CORS_ORIGINS env var or fallback to wildcard
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------
class ResumeRequest(BaseModel):
    resume_text: str
    target_role: str

class GitHubAnalyzeRequest(BaseModel):
    username: str
    target_role: str = ""

class InterviewStartRequest(BaseModel):
    candidate_profile: Optional[CandidateProfile] = None
    subject: str = "Python"
    thread_id: Optional[str] = None

class InterviewAnswerRequest(BaseModel):
    thread_id: str
    answer: str

class InterviewFinishRequest(BaseModel):
    thread_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _serialize_messages(messages: List[Any]) -> List[Dict[str, str]]:
    """Convert LangChain message objects to serializable role/content dicts."""
    serialized = []
    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type == "HumanMessage":
            serialized.append({"role": "user", "content": msg.content})
        elif msg_type == "AIMessage":
            serialized.append({"role": "assistant", "content": msg.content})
        elif msg_type == "SystemMessage":
            serialized.append({"role": "system", "content": msg.content})
        elif msg_type == "ToolMessage":
            serialized.append({"role": "tool", "content": str(msg.content)})
        else:
            serialized.append({"role": "unknown", "content": str(msg.content)})
    return serialized


def _extract_interview_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract serializable interview data from an agent invoke result."""
    messages = result.get("messages", [])
    
    last_ai_msg = ""
    for msg in reversed(messages):
        if type(msg).__name__ == "AIMessage":
            last_ai_msg = msg.content
            break
            
    return {
        "thread_id": None,  # filled by caller
        "question": last_ai_msg,
        "next_question": last_ai_msg,
        "messages": _serialize_messages(messages),
        "difficulty": result.get("difficulty", 2),
        "evaluation_rating": result.get("evaluation_rating"),
        "decision_log": result.get("decision_log", []),
    }


def _get_state_data(app, config) -> Dict[str, Any]:
    """Retrieve and serialize current agent state."""
    state = app.get_state(config)
    values = state.values if hasattr(state, "values") else state
    return {
        "messages": _serialize_messages(values.get("messages", [])),
        "candidate_profile": values.get("candidate_profile"),
        "subject": values.get("subject", "Python"),
        "difficulty": values.get("difficulty", 2),
        "evaluation_rating": values.get("evaluation_rating"),
        "decision_log": values.get("decision_log", []),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check."""
    return {"status": "healthy", "service": "PlaceMate AI API"}


# --- 1. Resume Analysis ---

@app.post("/api/candidate/analyze-resume")
def analyze_resume_endpoint(request: ResumeRequest):
    """
    Analyze a resume and extract a structured CandidateProfile.

    Calls the existing backend/parser.py::analyze_resume() function.
    """
    if not request.resume_text or not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty.")
    if not request.target_role or not request.target_role.strip():
        raise HTTPException(status_code=400, detail="target_role cannot be empty.")

    try:
        profile = analyze_resume(
            resume_text=request.resume_text.strip(),
            target_role=request.target_role.strip(),
        )
        return profile.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Resume analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Resume analysis failed due to a backend error.")


# --- 2. GitHub Analysis ---

@app.post("/api/github/analyze")
def analyze_github_endpoint(request: GitHubAnalyzeRequest):
    """
    Analyze a candidate's GitHub profile and repositories.

    Calls the existing backend/github/analyzer.py::fetch_and_analyze_github_profile().
    GitHub tokens are never exposed to the frontend — they come from environment only.
    """
    if not request.username or not request.username.strip():
        raise HTTPException(status_code=400, detail="username is required.")

    try:
        profile = fetch_and_analyze_github_profile(
            username=request.username.strip(),
            target_role=request.target_role.strip() if request.target_role else "",
        )
        if profile is None:
            raise HTTPException(status_code=500, detail="Failed to fetch or analyze GitHub profile.")
        return profile.model_dump()
    except Exception as e:
        logger.error(f"GitHub analysis failed: {e}")
        raise HTTPException(status_code=500, detail="GitHub analysis failed due to a backend error.")


# --- 3. Start Interview ---

@app.post("/api/interview/start")
def start_interview_endpoint(request: InterviewStartRequest):
    """
    Start a new interview session.

    Creates the LangGraph agent via build_interview_agent(), invokes it with
    the initial greeting message, and registers the session.
    """
    if not request.subject or not request.subject.strip():
        raise HTTPException(status_code=400, detail="subject is required.")

    subject = request.subject.strip()
    thread_id = request.thread_id or uuid.uuid4().hex

    try:
        from backend.agent import build_interview_agent

        candidate_profile = request.candidate_profile  # Optional[CandidateProfile] or None
        agent_app = build_interview_agent(subject=subject, candidate_profile=candidate_profile)
        config = get_session_config(thread_id)

        # Build the initial user message
        if candidate_profile:
            initial_msg = (
                f"Hello! I am {candidate_profile.name}, and I am ready to begin my "
                f"mock technical interview for the role of {candidate_profile.target_role} "
                f"focusing on {subject}."
            )
        else:
            initial_msg = f"Hello! I am ready to begin my mock interview on {subject}."

        result = agent_app.invoke(
            {"messages": [HumanMessage(content=initial_msg)], "difficulty": 2, "subject": subject, "candidate_profile": candidate_profile},
            config=config,
        )

        # Register session
        session_manager.create(
            thread_id=thread_id,
            app=agent_app,
            config=config,
            candidate_profile=candidate_profile,
            subject=subject,
        )

        interview_result = _extract_interview_result(result)
        interview_result["thread_id"] = thread_id
        return interview_result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Interview start failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to start interview.")


# --- 4. Submit Answer ---

@app.post("/api/interview/answer")
def answer_endpoint(request: InterviewAnswerRequest):
    """
    Submit a candidate answer and get the next interview question.

    Resumes the existing LangGraph session using the same thread_id.
    """
    if not request.thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required.")
    if not request.answer or not request.answer.strip():
        raise HTTPException(status_code=400, detail="answer cannot be empty.")

    session = session_manager.get(request.thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No active interview session found for thread_id '{request.thread_id}'.")

    try:
        agent_app = session["app"]
        config = session["config"]

        result = agent_app.invoke(
            {"messages": [HumanMessage(content=request.answer.strip())]},
            config=config,
        )

        interview_result = _extract_interview_result(result)
        interview_result["thread_id"] = request.thread_id
        return interview_result

    except Exception as e:
        logger.error(f"Interview answer failed for thread {request.thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process answer.")


# --- 5. Finish Interview ---

@app.post("/api/interview/finish")
def finish_interview_endpoint(request: InterviewFinishRequest):
    """
    Finish the interview and generate the FinalResult.

    Retrieves current graph state, extracts decision log and candidate profile,
    then calls the existing backend/results.py::generate_final_result().
    """
    if not request.thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required.")

    session = session_manager.get(request.thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No active interview session found for thread_id '{request.thread_id}'.")

    try:
        agent_app = session["app"]
        config = session["config"]

        # Get current state
        state_data = _get_state_data(agent_app, config)
        candidate_profile = state_data.get("candidate_profile")
        decision_log = state_data.get("decision_log", [])

        # Generate final result
        final_result = generate_final_result(
            candidate_profile=candidate_profile,
            decision_log=decision_log,
        )

        if final_result is None:
            raise HTTPException(status_code=500, detail="Failed to generate final result.")

        # Cache the result in the session
        session_manager.set_final_result(request.thread_id, final_result)

        return final_result.model_dump()

    except Exception as e:
        logger.error(f"Interview finish failed for thread {request.thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to finish interview.")


# --- 6. Get Interview Result ---

@app.get("/api/interview/result")
def interview_result_endpoint(thread_id: str = Query(..., description="The interview session thread ID.")):
    """
    Return the current interview state.

    Calls agent.get_state(config) to retrieve all messages, decisions, difficulty, etc.
    """
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required.")

    session = session_manager.get(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No active interview session found for thread_id '{thread_id}'.")

    try:
        # If a final result was already generated and cached, return it
        final_result = session_manager.get_final_result(thread_id)
        if final_result:
            result_dict = final_result.model_dump()
            result_dict["thread_id"] = thread_id
            return result_dict

        # Otherwise return the raw state data
        agent_app = session["app"]
        config = session["config"]

        state_data = _get_state_data(agent_app, config)
        state_data["thread_id"] = thread_id
        return state_data

    except Exception as e:
        logger.error(f"Failed to get interview result for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interview result.")


# --- 7. Get Recommendations ---

@app.get("/api/recommendations")
def recommendations_endpoint(thread_id: str = Query(..., description="The interview session thread ID.")):
    """
    Return job recommendations from the generated final result.

    Returns cached FinalResult if available (from finish), otherwise generates
    a new one from the current decision log.
    """
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required.")

    session = session_manager.get(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No active interview session found for thread_id '{thread_id}'.")

    try:
        # Check for cached final result first
        final_result = session_manager.get_final_result(thread_id)

        if final_result is None:
            # Generate from current state
            agent_app = session["app"]
            config = session["config"]
            state_data = _get_state_data(agent_app, config)
            candidate_profile = state_data.get("candidate_profile")
            decision_log = state_data.get("decision_log", [])
            final_result = generate_final_result(candidate_profile, decision_log)

            if final_result is None:
                raise HTTPException(status_code=500, detail="Failed to generate recommendations.")

            session_manager.set_final_result(thread_id, final_result)

        return {
            "thread_id": thread_id,
            "recommended_roles": final_result.recommended_roles,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommendations for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recommendations.")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    return {"error": str(exc)}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception at {request.url}: {exc}")
    return {"error": "An unexpected error occurred."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.api:app", host="0.0.0.0", port=port)
