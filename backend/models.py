"""
Data models for PlacementPrep AI — Milestones 2 & 3.

Defines Pydantic schemas for:
- Structured resume extraction and interview personalization (Milestone 2)
- Answer evaluation and decision logging for Agentic RAG (Milestone 3)
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ProjectSummary(BaseModel):
    """Structured summary of a candidate's project."""
    name: str = Field(description="Name or title of the project")
    technologies: List[str] = Field(
        default_factory=list,
        description="Key technologies, frameworks, and programming languages used in the project",
    )
    description: str = Field(
        description="Brief summary of the project's purpose, functionality, and candidate's contribution"
    )


class CandidateProfile(BaseModel):
    """
    Structured candidate profile extracted from resume text and target role.
    Used by the interview agent to conduct a personalized, adaptive interview.
    """
    name: str = Field(
        default="Candidate",
        description="Full name of the candidate extracted from the resume",
    )
    target_role: str = Field(
        description="The target job role the candidate is interviewing for",
    )
    skills: List[str] = Field(
        default_factory=list,
        description="Technical skills, languages, libraries, and tools mentioned in the resume",
    )
    projects: List[ProjectSummary] = Field(
        default_factory=list,
        description="List of structured project summaries extracted from the resume",
    )
    education: List[str] = Field(
        default_factory=list,
        description="Educational qualifications, degrees, institutions, and graduation years",
    )
    experience: List[str] = Field(
        default_factory=list,
        description="Past work experience, internships, or relevant technical roles",
    )
    interview_focus: List[str] = Field(
        default_factory=list,
        description="Core technical topics to prioritize during the interview based on target role requirements and candidate background",
    )
    potential_weak_areas: List[str] = Field(
        default_factory=list,
        description="Topics or concepts that should be probed deeper during the interview to verify competence (e.g. technologies listed without detailed projects, system design concepts, or trade-offs), not assumed to be confirmed weaknesses",
    )


class AnswerEvaluation(BaseModel):
    """
    Structured evaluation of a candidate's response to an interview question.
    Used to deterministically update interview difficulty.
    """
    rating: Literal["strong", "acceptable", "weak"] = Field(
        description="Evaluation rating: 'strong' (accurate, complete, shows depth), 'acceptable' (correct basics, partially complete), or 'weak' (inaccurate, major misconceptions, or superficial)"
    )
    feedback: str = Field(
        description="Concise 1-2 sentence assessment of the candidate's answer"
    )


class DecisionLogEntry(BaseModel):
    """
    Structured log record of the agent's decision process on a single turn.
    Exposed for UI display and live demonstration.
    """
    turn: int = Field(description="Turn number in the interview")
    retrieval_called: bool = Field(default=False, description="Whether the retrieval tool was invoked")
    retrieval_query: Optional[str] = Field(default=None, description="Query passed to the retrieval tool if invoked")
    retrieval_confidence: Optional[float] = Field(default=None, description="Similarity or confidence score of the retrieval")
    retrieval_score_gap: Optional[float] = Field(default=None, description="Difference between retrieval confidence and threshold (confidence - threshold)")
    retrieval_used: bool = Field(default=False, description="Whether retrieved material was accepted and used")
    fallback_used: bool = Field(default=False, description="Whether fallback to profile/history was triggered")
    difficulty_before: int = Field(default=2, description="Difficulty level before evaluating this turn (1-5)")
    difficulty_after: int = Field(default=2, description="Difficulty level after deterministic evaluation (1-5)")
    difficulty_delta: int = Field(default=0, description="Change in difficulty (difficulty_after - difficulty_before)")
    decision_reason: str = Field(default="", description="Concise rationale for difficulty adjustment and retrieval decision")
    evaluation_rating: Optional[str] = Field(default=None, description="Answer evaluation rating: strong, acceptable, or weak")

