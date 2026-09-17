import logging
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from backend.models import CandidateProfile, FinalResult

logger = logging.getLogger(__name__)

FINAL_RESULT_SYSTEM_PROMPT = """You are an expert technical interviewer and career counselor for PlacementPrep AI.
Your task is to review the candidate's profile and their interview decision log, and generate a comprehensive final evaluation report.

Candidate Profile:
Name: {name}
Target Role: {target_role}
Skills: {skills}

Interview Decision Log (Turns):
{decision_log}

Instructions:
1. Generate an objective score (0-100) for overall_score, technical_knowledge, problem_solving, communication, and confidence based on their performance in the interview.
2. If project-specific questions were asked and evaluated (check project_understanding in the log), provide a project_understanding score (0-100). Otherwise, leave it null.
3. Identify 2-4 key strengths.
4. Identify 2-4 weaknesses or areas for improvement.
5. Recommend 2-4 specific technical topics to study.
6. Provide 2-3 recommended job roles based on their skills, projects, and interview performance, including a match_score (0-100) and brief reasons.
"""

def generate_final_result(
    candidate_profile: Optional[CandidateProfile],
    decision_log: List[dict],
    llm: Optional[object] = None,
) -> Optional[FinalResult]:
    """
    Generate a structured final interview result and job recommendations.
    """
    if llm is None:
        from backend.agent import create_interviewer_llm
        llm = create_interviewer_llm()
        
    if type(llm).__name__ == "ChatOpenAI":
        structured_llm = llm.with_structured_output(FinalResult, method="function_calling")
    else:
        structured_llm = llm.with_structured_output(FinalResult)
    
    log_str = ""
    for entry in decision_log:
        log_str += f"Turn {entry.get('turn')}: Difficulty {entry.get('difficulty_before')}->{entry.get('difficulty_after')}. Rating: {entry.get('evaluation_rating')}. Reason: {entry.get('decision_reason')}\n"

    system_prompt = FINAL_RESULT_SYSTEM_PROMPT.format(
        name=candidate_profile.name if candidate_profile else "Candidate",
        target_role=candidate_profile.target_role if candidate_profile else "Unknown",
        skills=", ".join(candidate_profile.skills) if candidate_profile and candidate_profile.skills else "None",
        decision_log=log_str
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please generate the final evaluation report.")
    ]
    
    try:
        final_result = structured_llm.invoke(messages)
        
        # Reconstruct the evaluations array from the decision_log
        from backend.models import EvaluationRecord
        evaluations = []
        for entry in decision_log:
            q_text = entry.get('question_text')
            c_ans = entry.get('candidate_answer')
            if q_text and c_ans:
                evaluations.append(EvaluationRecord(
                    question_text=q_text,
                    candidate_answer=c_ans,
                    rating=entry.get('evaluation_rating') or 'unknown',
                    feedback=entry.get('feedback') or 'No feedback provided.',
                    difficulty=entry.get('difficulty_after', 2)
                ))
        
        final_result.evaluations = evaluations
        return final_result
    except Exception as e:
        logger.error(f"Failed to generate final result: {e}")
        return None
