"""
Answer evaluation and deterministic difficulty management for PlacementPrep AI.

Milestone 3:
- Evaluates whether the candidate's answer addressed the interview question
- Uses structured output (AnswerEvaluation: strong, acceptable, weak)
- Deterministically updates interview difficulty (scale 1 to 5)
- Does NOT let the LLM freely assign arbitrary difficulty numbers
"""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from backend.models import AnswerEvaluation

EVALUATION_SYSTEM_PROMPT = """You are an objective technical interview evaluator for PlacementPrep AI.
Your job is to assess a candidate's answer to a technical interview question on {subject}.

Evaluation Criteria:
- 'strong': The answer is technically accurate, complete, demonstrates depth of understanding, and addresses trade-offs or mechanisms correctly.
- 'acceptable': The answer is fundamentally correct on basic concepts, but lacks depth, misses minor nuances, or is somewhat brief.
- 'weak': The answer contains significant technical inaccuracies, shows major confusion or knowledge gaps, or fails to address the question asked.

Provide an objective rating ('strong', 'acceptable', or 'weak') and a concise 1-2 sentence feedback.
"""


def compute_next_difficulty(current_difficulty: int, rating: str) -> int:
    """
    Deterministically calculate the next difficulty level based on answer rating.
    Difficulty is bounded between 1 (foundational) and 5 (architectural/deep-dive).

    Rules:
    - strong: difficulty + 1 (capped at 5)
    - acceptable: difficulty unchanged
    - weak: difficulty - 1 (floored at 1)
    """
    clean_rating = (rating or "").strip().lower()

    if clean_rating == "strong":
        return min(5, current_difficulty + 1)
    elif clean_rating == "weak":
        return max(1, current_difficulty - 1)
    else:  # "acceptable" or unrecognized
        return max(1, min(5, current_difficulty))


def evaluate_candidate_answer(
    question: str,
    candidate_answer: str,
    subject: str = "Python",
    llm: Optional[object] = None,
) -> AnswerEvaluation:
    """
    Evaluate a candidate's response using structured output.

    Args:
        question: The interview question that was asked.
        candidate_answer: The answer provided by the candidate.
        subject: The interview technical topic (default: "Python").
        llm: Optional chat model. If None, initializes Gemini 3.5 Flash.

    Returns:
        An AnswerEvaluation Pydantic object.
    """
    if not candidate_answer or not candidate_answer.strip():
        return AnswerEvaluation(
            rating="weak",
            feedback="Candidate did not provide an answer.",
        )

    if llm is None:
        from backend.agent import create_interviewer_llm
        llm = create_interviewer_llm()

    structured_evaluator = llm.with_structured_output(AnswerEvaluation)

    system_prompt = EVALUATION_SYSTEM_PROMPT.format(subject=subject)
    user_prompt = f"Interview Question: {question.strip()}\n\nCandidate Answer: {candidate_answer.strip()}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    evaluation = structured_evaluator.invoke(messages)
    return evaluation
