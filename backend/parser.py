"""
Resume parser and profile analyzer for PlacementPrep AI.

Milestone 2: Extracts structured candidate information from resume text
and generates a validated CandidateProfile using Google Gemini 3.5 Flash.
"""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from backend.models import CandidateProfile

RESUME_PARSER_SYSTEM_PROMPT = """You are an expert technical recruitment analyst for PlacementPrep AI.
Your job is to analyze candidate resume text and extract a structured Candidate Profile for a mock technical interview.

Target Role: {target_role}

Instructions:
1. Fact-based Extraction:
   - Extract the candidate's name, technical skills, projects, education, and experience strictly from the provided resume text.
   - For each project, extract the project name, technologies used, and a concise 1-2 sentence description.
   - Do not invent or assume details not present in the resume text.

2. Strategic Interview Assessment:
   - interview_focus: Identify 3 to 5 core technical topics or competencies most relevant to the target role ({target_role}) and the candidate's stated background that should be prioritized during the interview.
   - potential_weak_areas: Identify 2 to 4 technical concepts, gaps, or trade-offs that the interviewer should actively PROBE during the interview to verify depth of knowledge (e.g. technologies listed without detailed projects, concurrency concepts, scaling trade-offs, testing methodologies).
   - IMPORTANT: Treat potential_weak_areas solely as areas to verify and probe during the conversation, NOT as confirmed weaknesses.
"""


def analyze_resume(
    resume_text: str,
    target_role: str,
    llm: Optional[object] = None,
) -> CandidateProfile:
    """
    Analyze raw resume text and extract a structured, validated CandidateProfile.

    Args:
        resume_text: Raw plain text of the candidate's resume.
        target_role: The target job position (e.g. 'Junior Python Backend Developer').
        llm: Optional chat model. If None, initializes Gemini 3.5 Flash.

    Returns:
        A validated CandidateProfile Pydantic object.

    Raises:
        ValueError: If resume_text or target_role is empty or whitespace only.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    if not target_role or not target_role.strip():
        raise ValueError("Target role cannot be empty.")

    clean_resume = resume_text.strip()
    clean_role = target_role.strip()

    if llm is None:
        from backend.agent import create_interviewer_llm
        llm = create_interviewer_llm()

    # Configure structured output bound to the CandidateProfile schema
    structured_llm = llm.with_structured_output(CandidateProfile)

    system_prompt = RESUME_PARSER_SYSTEM_PROMPT.format(target_role=clean_role)
    user_prompt = f"Target Role: {clean_role}\n\nCandidate Resume:\n{clean_resume}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    profile = structured_llm.invoke(messages)

    # Ensure target_role is set properly on the returned profile
    if not profile.target_role or profile.target_role.strip() == "":
        profile.target_role = clean_role

    return profile
