"""
Tests for the PlaceMate AI REST API layer (backend/api.py).

Covers:
1. Resume analysis endpoint
2. GitHub analysis endpoint with mocked GitHub client
3. Interview start endpoint
4. Interview answer endpoint
5. Invalid thread_id handling
6. Interview finish endpoint
7. Result endpoint
8. Recommendations endpoint

All LLM/GitHub calls are mocked so tests run without live API keys.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Ensure backend is importable
from backend.api import app
from backend.models import CandidateProfile, FinalResult, GitHubProfile, JobRecommendation, ProjectProfile, ProjectSummary


client = TestClient(app)
from backend.session_manager import session_manager


# ---------------------------------------------------------------------------
# Mock app for interview endpoints
# ---------------------------------------------------------------------------
class MockInterviewApp:
    """
    A mock LangGraph app that mimics build_interview_agent().
    - invoke() returns a predefined response and decision_log
    - get_state() returns the current session state
    """

    def __init__(self, responses=None):
        self.responses = responses or ["Welcome! First question: What is Python?"]
        self.call_count = 0
        self._state = {
            "messages": [],
            "difficulty": 2,
            "decision_log": [],
            "candidate_profile": None,
            "subject": "Python",
            "evaluation_rating": None,
        }

    def invoke(self, input_dict, config=None):
        user_msgs = [m for m in input_dict.get("messages", []) if type(m).__name__ == "HumanMessage"]
        if user_msgs:
            self._state["messages"].append(user_msgs[-1])

        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = "Thank you. Interview complete."
        self.call_count += 1
        ai_msg = AIMessage(content=resp)
        self._state["messages"].append(ai_msg)

        # Record a decision log entry on turns after the first
        if self.call_count > 1:
            self._state["decision_log"].append({
                "turn": self.call_count,
                "evaluation_rating": "acceptable",
                "difficulty_before": 2,
                "difficulty_after": 2,
                "retrieval_called": False,
                "retrieval_used": False,
                "fallback_used": False,
                "decision_reason": "Mock decision.",
            })

        return {
            "messages": list(self._state["messages"]),
            "difficulty": self._state["difficulty"],
            "decision_log": list(self._state["decision_log"]),
            "evaluation_rating": self._state["evaluation_rating"],
        }

    def get_state(self, config=None):
        class StateObj:
            def __init__(self, values):
                self.values = values
        return StateObj(dict(self._state))


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear session manager between tests."""
    from backend.session_manager import session_manager
    for tid in list(session_manager.list_threads()):
        session_manager.delete(tid)
    yield
    for tid in list(session_manager.list_threads()):
        session_manager.delete(tid)


def sample_candidate_profile():
    return CandidateProfile(
        name="Alex Chen",
        target_role="Junior Python Backend Developer",
        skills=["Python", "Django", "FastAPI"],
        projects=[
            ProjectSummary(name="Log Aggregator", technologies=["FastAPI", "Redis"], description="Log ingestion service.")
        ],
        education=["B.Tech CSE"],
        experience=["Intern at CloudScale"],
        interview_focus=["Async programming", "REST APIs"],
        potential_weak_areas=["System design"],
    )


# ---------------------------------------------------------------------------
# 1. Resume analysis
# ---------------------------------------------------------------------------
class TestAnalyzeResume:
    def test_success(self):
        profile = sample_candidate_profile()
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock(invoke=MagicMock(return_value=profile))

        with patch("backend.api.analyze_resume", return_value=profile):
            response = client.post(
                "/api/candidate/analyze-resume",
                json={"resume_text": "Alex Chen — Python, Django — Project: Log Aggregator", "target_role": "Backend Developer"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Alex Chen"
        assert data["target_role"] == "Junior Python Backend Developer"

    def test_missing_resume_text(self):
        response = client.post(
            "/api/candidate/analyze-resume",
            json={"resume_text": "", "target_role": "Backend Developer"},
        )
        assert response.status_code == 400

    def test_missing_target_role(self):
        response = client.post(
            "/api/candidate/analyze-resume",
            json={"resume_text": "Some resume text", "target_role": ""},
        )
        assert response.status_code == 400

    def test_llm_error_returns_500(self):
        with patch("backend.api.analyze_resume", side_effect=RuntimeError("LLM error")):
            response = client.post(
                "/api/candidate/analyze-resume",
                json={"resume_text": "Some resume text", "target_role": "Backend Developer"},
            )
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# 2. GitHub analysis
# ---------------------------------------------------------------------------
class TestAnalyzeGitHub:
    def test_success(self):
        mock_profile = GitHubProfile(
            username="testuser",
            profile_url="https://github.com/testuser",
            projects=[],
        )
        with patch("backend.api.fetch_and_analyze_github_profile", return_value=mock_profile):
            response = client.post(
                "/api/github/analyze",
                json={"username": "testuser", "target_role": "Backend Developer"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"

    def test_missing_username(self):
        response = client.post("/api/github/analyze", json={"username": ""})
        assert response.status_code == 400

    def test_failure_returns_500(self):
        with patch("backend.api.fetch_and_analyze_github_profile", return_value=None):
            response = client.post(
                "/api/github/analyze",
                json={"username": "nonexistent_user"},
            )
        assert response.status_code == 500

    def test_no_token_exposed(self):
        """Verify the endpoint never accepts or returns a GitHub token."""
        mock_profile = GitHubProfile(username="testuser", profile_url="https://github.com/testuser", projects=[])
        with patch("backend.api.fetch_and_analyze_github_profile", return_value=mock_profile) as mock_fn:
            response = client.post(
                "/api/github/analyze",
                json={"username": "testuser", "target_role": "Backend", "token": "ghp_secret_token"},
            )
        # The API does not pass the token to fetch_and_analyze_github_profile
        call_kwargs = mock_fn.call_args[1]
        assert "token" not in call_kwargs


# ---------------------------------------------------------------------------
# 3. Interview start
# ---------------------------------------------------------------------------
class TestInterviewStart:
    def test_start_with_profile(self):
        profile = sample_candidate_profile()
        mock_app = MockInterviewApp(responses=["Welcome, Alex! Question 1: What is Python?"])

        with patch("backend.agent.build_interview_agent", return_value=mock_app):
            response = client.post(
                "/api/interview/start",
                json={"candidate_profile": profile.model_dump(), "subject": "Python"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert len(data["messages"]) > 0
        assert data["difficulty"] == 2
        assert "decision_log" in data
        # Session should be registered
        assert session_manager.has(data["thread_id"])

    def test_start_without_profile(self):
        mock_app = MockInterviewApp(responses=["Welcome! Question 1: What is Python?"])
        with patch("backend.agent.build_interview_agent", return_value=mock_app):
            response = client.post(
                "/api/interview/start",
                json={"subject": "Python"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert session_manager.has(data["thread_id"])

    def test_start_missing_subject(self):
        response = client.post(
            "/api/interview/start",
            json={"subject": ""},
        )
        assert response.status_code == 400

    def test_start_llm_error_returns_500(self):
        with patch("backend.agent.build_interview_agent", side_effect=RuntimeError("LLM init failed")):
            response = client.post(
                "/api/interview/start",
                json={"subject": "Python"},
            )
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# 4. Interview answer
# ---------------------------------------------------------------------------
class TestInterviewAnswer:
    def test_submit_answer(self):
        profile = sample_candidate_profile()
        mock_app = MockInterviewApp(responses=["Good answer! Question 2: Explain GIL."])

        with patch("backend.agent.build_interview_agent", return_value=mock_app):
            start_resp = client.post(
                "/api/interview/start",
                json={"candidate_profile": profile.model_dump(), "subject": "Python"},
            )
        thread_id = start_resp.json()["thread_id"]

        # Submit answer
        answer_resp = client.post(
            "/api/interview/answer",
            json={"thread_id": thread_id, "answer": "Python is dynamically typed."},
        )
        assert answer_resp.status_code == 200
        data = answer_resp.json()
        assert data["thread_id"] == thread_id
        assert len(data["messages"]) > 0
        assert "decision_log" in data

    def test_empty_answer_returns_400(self):
        response = client.post(
            "/api/interview/answer",
            json={"thread_id": "unknown", "answer": ""},
        )
        assert response.status_code == 400

    def test_unknown_thread_id_returns_404(self):
        response = client.post(
            "/api/interview/answer",
            json={"thread_id": "nonexistent-thread", "answer": "My answer"},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# 5. Interview finish
# ---------------------------------------------------------------------------
class TestInterviewFinish:
    def test_finish_returns_final_result(self):
        profile = sample_candidate_profile()
        mock_app = MockInterviewApp(responses=["Question 1.", "Question 2."])

        with patch("backend.agent.build_interview_agent", return_value=mock_app):
            start_resp = client.post(
                "/api/interview/start",
                json={"candidate_profile": profile.model_dump(), "subject": "Python"},
            )
        thread_id = start_resp.json()["thread_id"]

        # Submit an answer first
        client.post("/api/interview/answer", json={"thread_id": thread_id, "answer": "My answer."})

        # Mock generate_final_result
        final_result = FinalResult(
            overall_score=85,
            technical_knowledge=90,
            problem_solving=80,
            communication=85,
            confidence=75,
            strengths=["Strong fundamentals", "Good communication"],
            weaknesses=["Needs more system design practice"],
            recommended_topics=["System design"],
            recommended_roles=[JobRecommendation(role="Backend Developer", match_score=88, reasons=["Skills match"])],
        )
        with patch("backend.api.generate_final_result", return_value=final_result):
            finish_resp = client.post(
                "/api/interview/finish",
                json={"thread_id": thread_id},
            )
        assert finish_resp.status_code == 200
        data = finish_resp.json()
        assert data["overall_score"] == 85
        assert len(data["recommended_roles"]) > 0

        # Verify cached in session
        cached = session_manager.get_final_result(thread_id)
        assert cached is not None
        assert cached.overall_score == 85

    def test_finish_unknown_thread(self):
        response = client.post("/api/interview/finish", json={"thread_id": "nonexistent"})
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# 6. Interview result
# ---------------------------------------------------------------------------
class TestInterviewResult:
    def test_get_result(self):
        profile = sample_candidate_profile()
        mock_app = MockInterviewApp(responses=["Question 1."])

        with patch("backend.agent.build_interview_agent", return_value=mock_app):
            start_resp = client.post(
                "/api/interview/start",
                json={"candidate_profile": profile.model_dump(), "subject": "Python"},
            )
        thread_id = start_resp.json()["thread_id"]

        response = client.get(f"/api/interview/result?thread_id={thread_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == thread_id
        assert "messages" in data
        assert "decision_log" in data
        assert data["difficulty"] == 2

    def test_result_unknown_thread(self):
        response = client.get("/api/interview/result?thread_id=nonexistent")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# 7. Recommendations
# ---------------------------------------------------------------------------
class TestRecommendations:
    def test_get_recommendations(self):
        profile = sample_candidate_profile()
        mock_app = MockInterviewApp(responses=["Question 1."])

        with patch("backend.agent.build_interview_agent", return_value=mock_app):
            start_resp = client.post(
                "/api/interview/start",
                json={"candidate_profile": profile.model_dump(), "subject": "Python"},
            )
        thread_id = start_resp.json()["thread_id"]

        final_result = FinalResult(
            overall_score=80,
            technical_knowledge=85,
            problem_solving=75,
            communication=80,
            confidence=70,
            strengths=["Good skills"],
            weaknesses=["Needs more practice"],
            recommended_topics=["Testing"],
            recommended_roles=[
                JobRecommendation(role="Backend Developer", match_score=82, reasons=["Match"]),
                JobRecommendation(role="Python Developer", match_score=78, reasons=["Python skills"]),
            ],
        )
        with patch("backend.api.generate_final_result", return_value=final_result):
            response = client.get(f"/api/recommendations?thread_id={thread_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == thread_id
        assert len(data["recommended_roles"]) == 2
        assert data["recommended_roles"][0]["match_score"] == 82

    def test_recommendations_unknown_thread(self):
        response = client.get("/api/recommendations?thread_id=nonexistent")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# 8. Health check
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
