"""
Session manager for PlaceMate AI API.

Maps thread_id to interview agent, configuration, candidate profile, and
cached final result across HTTP requests.
"""

import threading
from typing import Any, Dict, List, Optional

from backend.models import CandidateProfile, FinalResult


class SessionManager:
    """
    Thread-safe registry of active interview sessions.

    Maps thread_id -> session dict containing:
      - app: compiled LangGraph interview agent
      - config: LangGraph session config (with thread_id)
      - candidate_profile: Optional[CandidateProfile]
      - subject: str
      - final_result: Optional[FinalResult] (cached after finish)
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(
        self,
        thread_id: str,
        app: Any,
        config: Dict[str, Any],
        candidate_profile: Optional[CandidateProfile],
        subject: str,
    ) -> None:
        """Register a new interview session."""
        with self._lock:
            self._sessions[thread_id] = {
                "app": app,
                "config": config,
                "candidate_profile": candidate_profile,
                "subject": subject,
                "final_result": None,
            }

    def get(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Return session data for the given thread_id, or None."""
        with self._lock:
            return self._sessions.get(thread_id)

    def has(self, thread_id: str) -> bool:
        """Check whether a session exists for the given thread_id."""
        with self._lock:
            return thread_id in self._sessions

    def delete(self, thread_id: str) -> None:
        """Remove a session."""
        with self._lock:
            self._sessions.pop(thread_id, None)

    def set_final_result(self, thread_id: str, result: FinalResult) -> None:
        """Cache a FinalResult for the session."""
        with self._lock:
            if thread_id in self._sessions:
                self._sessions[thread_id]["final_result"] = result

    def get_final_result(self, thread_id: str) -> Optional[FinalResult]:
        """Return cached FinalResult, or None."""
        with self._lock:
            session = self._sessions.get(thread_id)
            if session is None:
                return None
            return session.get("final_result")

    def list_threads(self) -> List[str]:
        """Return all active thread IDs."""
        with self._lock:
            return list(self._sessions.keys())


# Module-level singleton
session_manager = SessionManager()
