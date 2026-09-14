from uuid import uuid4

from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_core.messages import HumanMessage

from backend.agent import build_interview_agent, extract_message_text
from backend.memory import create_memory_saver, get_session_config


app = Flask(__name__)
CORS(app)


# One shared checkpointer keeps interview conversations available
# across API requests. Each interview gets its own thread_id.
memory = create_memory_saver()

_sessions = {}


def _get_agent(subject):
    """Build the existing Person 1 LangGraph agent."""
    return build_interview_agent(
        subject=subject,
        checkpointer=memory,
    )


def _latest_ai_message(result):
    """Extract the latest AI response from LangGraph state."""
    messages = result.get("messages", [])

    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            return extract_message_text(message.content)

    return ""


@app.get("/api/health")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "PlacementPrep-AI API",
    })


@app.post("/api/interview/start")
def start_interview():
    data = request.get_json(silent=True) or {}

    subject = str(data.get("subject", "Python")).strip()

    if not subject:
        return jsonify({
            "error": "subject is required",
        }), 400

    thread_id = str(uuid4())

    agent = _get_agent(subject)
    config = get_session_config(thread_id)

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        f"Hello! I am ready to begin my mock interview "
                        f"on {subject}."
                    )
                )
            ]
        },
        config=config,
    )

    first_question = _latest_ai_message(result)

    _sessions[thread_id] = {
        "subject": subject,
        "agent": agent,
    }

    return jsonify({
        "thread_id": thread_id,
        "subject": subject,
        "question": first_question,
    }), 201


@app.post("/api/interview/answer")
def submit_answer():
    data = request.get_json(silent=True) or {}

    thread_id = str(data.get("thread_id", "")).strip()
    answer = str(data.get("answer", "")).strip()

    if not thread_id:
        return jsonify({
            "error": "thread_id is required",
        }), 400

    if not answer:
        return jsonify({
            "error": "answer is required",
        }), 400

    session = _sessions.get(thread_id)

    if session is None:
        return jsonify({
            "error": "interview session not found",
        }), 404

    agent = session["agent"]
    config = get_session_config(thread_id)

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=answer),
            ]
        },
        config=config,
    )

    next_question = _latest_ai_message(result)

    return jsonify({
        "thread_id": thread_id,
        "question": next_question,
    })


@app.get("/api/interview/state")
def interview_state():
    thread_id = request.args.get("thread_id", "").strip()

    if not thread_id:
        return jsonify({
            "error": "thread_id is required",
        }), 400

    session = _sessions.get(thread_id)

    if session is None:
        return jsonify({
            "error": "interview session not found",
        }), 404

    config = get_session_config(thread_id)
    checkpoint = memory.get(config)

    messages = []

    if checkpoint:
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])

    return jsonify({
        "thread_id": thread_id,
        "subject": session["subject"],
        "message_count": len(messages),
    })


@app.post("/api/interview/end")
def end_interview():
    data = request.get_json(silent=True) or {}

    thread_id = str(data.get("thread_id", "")).strip()

    if not thread_id:
        return jsonify({
            "error": "thread_id is required",
        }), 400

    session = _sessions.get(thread_id)

    if session is None:
        return jsonify({
            "error": "interview session not found",
        }), 404

    config = get_session_config(thread_id)
    checkpoint = memory.get(config)

    message_count = 0

    if checkpoint:
        channel_values = checkpoint.get("channel_values", {})
        message_count = len(channel_values.get("messages", []))

    _sessions.pop(thread_id, None)

    return jsonify({
        "thread_id": thread_id,
        "subject": session["subject"],
        "status": "ended",
        "message_count": message_count,
    })


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )