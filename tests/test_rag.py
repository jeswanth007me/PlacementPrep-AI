"""
Unit and integration tests for Milestone 3: Agentic RAG & Decision Logic.

Covers:
1. Retrieval tool returns high-confidence results for relevant domain queries
2. Retrieval tool returns low-confidence results for out-of-scope queries
3. Retrieval tool handles empty queries gracefully
4. Deterministic difficulty calculation (strong increases, weak decreases, acceptable maintains)
5. Decision log structure and validation
6. Graph execution when agent chooses retrieval vs skips retrieval
"""

import json
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from backend.evaluation import compute_next_difficulty
from backend.models import AnswerEvaluation, DecisionLogEntry
from backend.rag import LocalPrepRetriever, retrieve_prep_material, cosine_similarity
from backend.memory import create_memory_saver, get_session_config
from backend.agent import build_interview_agent


class TestAgenticRAGAndEvaluation(unittest.TestCase):

    def test_cosine_similarity_edge_cases(self):
        """Verify cosine similarity calculation with identical, orthogonal, and zero vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        vec_zero = [0.0, 0.0, 0.0]

        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 1.0, places=4)
        self.assertAlmostEqual(cosine_similarity(vec1, vec3), 0.0, places=4)
        self.assertEqual(cosine_similarity(vec1, vec_zero), 0.0)

    def test_retrieval_empty_query(self):
        """Verify empty query returns EMPTY status with 0.0 confidence."""
        retriever = LocalPrepRetriever()
        res = retriever.retrieve("")
        self.assertEqual(res["status"], "EMPTY")
        self.assertEqual(res["confidence"], 0.0)
        self.assertEqual(res["chunks"], [])
        self.assertIn("Fall back", res["guidance"])

        res_ws = retriever.retrieve("   \n\t  ")
        self.assertEqual(res_ws["status"], "EMPTY")

    def test_retrieval_high_confidence_match(self):
        """Verify relevant domain queries achieve high confidence above threshold."""
        retriever = LocalPrepRetriever()
        res = retriever.retrieve("Redis persistence RDB AOF trade-offs")
        self.assertEqual(res["status"], "HIGH_CONFIDENCE")
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertTrue(len(res["chunks"]) > 0)
        self.assertIn("Redis", res["chunks"][0])
        self.assertIn("High-confidence", res["guidance"])

    def test_retrieval_low_confidence_fallback(self):
        """Verify out-of-scope queries produce low confidence and trigger fallback guidance."""
        retriever = LocalPrepRetriever()
        res = retriever.retrieve("ancient roman aqueduct engineering methods")
        self.assertEqual(res["status"], "LOW_CONFIDENCE")
        self.assertLess(res["confidence"], 0.70)
        self.assertEqual(res["chunks"], [])
        self.assertIn("DO NOT use external material", res["guidance"])
        self.assertIn("Fall back", res["guidance"])

    def test_deterministic_difficulty_scaling(self):
        """Verify difficulty rules strictly enforce 1-5 bounds and deterministic transitions."""
        # Strong answers increment difficulty
        self.assertEqual(compute_next_difficulty(1, "strong"), 2)
        self.assertEqual(compute_next_difficulty(2, "strong"), 3)
        self.assertEqual(compute_next_difficulty(4, "strong"), 5)
        self.assertEqual(compute_next_difficulty(5, "strong"), 5)  # Cap at 5

        # Weak answers decrement difficulty
        self.assertEqual(compute_next_difficulty(5, "weak"), 4)
        self.assertEqual(compute_next_difficulty(3, "weak"), 2)
        self.assertEqual(compute_next_difficulty(2, "weak"), 1)
        self.assertEqual(compute_next_difficulty(1, "weak"), 1)  # Floor at 1

        # Acceptable answers keep difficulty unchanged
        self.assertEqual(compute_next_difficulty(2, "acceptable"), 2)
        self.assertEqual(compute_next_difficulty(4, "acceptable"), 4)

    def test_decision_log_model_validation(self):
        """Verify DecisionLogEntry validates all 10 decision trace fields."""
        log_entry = DecisionLogEntry(
            turn=2,
            retrieval_called=True,
            retrieval_query="Redis persistence",
            retrieval_confidence=0.814,
            retrieval_score_gap=0.114,
            retrieval_used=True,
            fallback_used=False,
            difficulty_before=2,
            difficulty_after=3,
            difficulty_delta=1,
            decision_reason="Answer rated strong (difficulty 2->3, delta +1). Retrieved prep material with confidence 0.814 (gap +0.114 >= 0.0).",
            evaluation_rating="strong",
        )
        data = log_entry.model_dump()
        self.assertEqual(data["turn"], 2)
        self.assertTrue(data["retrieval_called"])
        self.assertEqual(data["retrieval_query"], "Redis persistence")
        self.assertEqual(data["retrieval_confidence"], 0.814)
        self.assertAlmostEqual(data["retrieval_score_gap"], 0.114, places=3)
        self.assertTrue(data["retrieval_used"])
        self.assertFalse(data["fallback_used"])
        self.assertEqual(data["difficulty_before"], 2)
        self.assertEqual(data["difficulty_after"], 3)
        self.assertEqual(data["difficulty_delta"], 1)
        self.assertIn("difficulty 2->3", data["decision_reason"])
        self.assertEqual(data["evaluation_rating"], "strong")

    def test_agent_skips_retrieval_flow(self):
        """Verify graph flows directly to evaluate_and_update when model does not call tools."""
        class MockDirectLLM:
            def __init__(self):
                self.call_count = 0
            def invoke(self, messages):
                self.call_count += 1
                return AIMessage(content="Can you explain how Python generators differ from lists?")

        mock_llm = MockDirectLLM()
        checkpointer = create_memory_saver()
        agent = build_interview_agent(subject="Python", llm=mock_llm, checkpointer=checkpointer)

        config = get_session_config("test-skip-retrieval")
        init_res = agent.invoke({"messages": [HumanMessage(content="Ready for interview")]}, config=config)

        # Confirm response generated and decision log recorded
        self.assertIn("generators", init_res["messages"][-1].content)
        self.assertEqual(len(init_res.get("decision_log", [])), 1)
        first_log = init_res["decision_log"][0]
        self.assertFalse(first_log["retrieval_called"])
        self.assertFalse(first_log["retrieval_used"])
        self.assertFalse(first_log["fallback_used"])
        self.assertIsNone(first_log["retrieval_score_gap"])
        self.assertEqual(first_log["difficulty_delta"], 0)
        self.assertEqual(first_log["difficulty_after"], 2)
        self.assertTrue(len(first_log["decision_reason"]) > 0)

    def test_agent_retrieval_tool_calling_flow(self):
        """Verify graph routes through ToolNode and back to interviewer when tool_calls are returned."""
        class MockToolCallingLLM:
            def __init__(self):
                self.call_count = 0
            def invoke(self, messages):
                self.call_count += 1
                if self.call_count == 1:
                    # Turn 1: Call the retrieval tool
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "retrieve_prep_material",
                            "args": {"query": "Redis persistence"},
                            "id": "call_test_123",
                        }]
                    )
                else:
                    # Turn 2: Received tool response, formulate follow-up question
                    return AIMessage(content="Based on Redis persistence, how does AOF rewrite work?")

        mock_llm = MockToolCallingLLM()
        checkpointer = create_memory_saver()
        agent = build_interview_agent(subject="Python", llm=mock_llm, checkpointer=checkpointer)

        config = get_session_config("test-tool-calling")
        res = agent.invoke({"messages": [HumanMessage(content="I am not fully sure about Redis persistence.")]}, config=config)

        # Verify final message is the follow-up question
        final_msg = res["messages"][-1]
        self.assertIsInstance(final_msg, AIMessage)
        self.assertIn("AOF rewrite", final_msg.content)

        # Verify a ToolMessage was produced during execution
        tool_messages = [m for m in res["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].name, "retrieve_prep_material")

        # Verify decision log recorded retrieval
        self.assertEqual(len(res.get("decision_log", [])), 1)
        log = res["decision_log"][0]
        self.assertTrue(log["retrieval_called"])
        self.assertEqual(log["retrieval_query"], "Redis persistence")
        self.assertIsNotNone(log["retrieval_confidence"])
        self.assertGreaterEqual(log["retrieval_confidence"], 0.70)
        self.assertTrue(log["retrieval_used"])
        self.assertFalse(log["fallback_used"])
        self.assertIsNotNone(log["retrieval_score_gap"])
        self.assertGreaterEqual(log["retrieval_score_gap"], 0.0)

    def test_agent_low_confidence_retrieval_fallback_flow(self):
        """
        Verify that when retrieval confidence is below threshold:
        1. Tool returns LOW_CONFIDENCE and empty chunks.
        2. Material is NOT passed to question generation.
        3. Agent falls back to candidate profile and conversation context.
        4. Decision log records fallback_used = True, retrieval_used = False,
           and negative retrieval_score_gap.
        """
        class MockLowConfidenceToolLLM:
            def __init__(self):
                self.call_count = 0
            def invoke(self, messages):
                self.call_count += 1
                if self.call_count == 1:
                    # Model decides to query out-of-scope knowledge
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "retrieve_prep_material",
                            "args": {"query": "medieval blacksmith forging temperature control"},
                            "id": "call_test_low_conf",
                        }]
                    )
                else:
                    # Receives LOW_CONFIDENCE result, sees guidance to fall back,
                    # and falls back to candidate profile/conversation context
                    return AIMessage(content="Let's focus back on your Python project experience with Redis caching.")

        mock_llm = MockLowConfidenceToolLLM()
        checkpointer = create_memory_saver()
        agent = build_interview_agent(subject="Python", llm=mock_llm, checkpointer=checkpointer)

        config = get_session_config("test-low-conf-fallback")
        res = agent.invoke({"messages": [HumanMessage(content="I want to ask about ancient forging techniques.")]}, config=config)

        # 1. Tool message check
        tool_messages = [m for m in res["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_messages), 1)
        tool_data = json.loads(tool_messages[0].content)
        self.assertEqual(tool_data["status"], "LOW_CONFIDENCE")
        self.assertEqual(tool_data["chunks"], [])
        self.assertLess(tool_data["confidence"], 0.70)

        # 2. Final response check (fallback)
        final_msg = res["messages"][-1]
        self.assertIn("Python project experience", final_msg.content)

        # 3. Decision log verification
        self.assertEqual(len(res.get("decision_log", [])), 1)
        log = res["decision_log"][0]
        self.assertTrue(log["retrieval_called"])
        self.assertEqual(log["retrieval_query"], "medieval blacksmith forging temperature control")
        self.assertLess(log["retrieval_confidence"], 0.70)
        self.assertFalse(log["retrieval_used"])
        self.assertTrue(log["fallback_used"])
        self.assertIsNotNone(log["retrieval_score_gap"])
        self.assertLess(log["retrieval_score_gap"], 0.0)
        self.assertIn("below threshold", log["decision_reason"])

    def test_agent_evaluation_failure_handling(self):
        """
        Verify that if the evaluator fails, the system safely falls back
        to 'error', does not change difficulty, and logs it correctly.
        """
        class FailingMockLLM:
            def invoke(self, messages):
                return AIMessage(content="Question: Next?")

        class FailingEvalLLM:
            def with_structured_output(self, schema):
                class FailingEvaluator:
                    def invoke(self, messages):
                        raise Exception("API Timeout Simulation")
                return FailingEvaluator()

        checkpointer = create_memory_saver()
        agent = build_interview_agent(
            subject="Python",
            llm=FailingMockLLM(),
            eval_llm=FailingEvalLLM(),
            checkpointer=checkpointer
        )

        config = get_session_config("test-eval-fail")

        # Turn 1: initialize
        agent.invoke({"messages": [HumanMessage(content="Hello!")]}, config=config)

        # Turn 2: answer
        res = agent.invoke({"messages": [HumanMessage(content="My answer.")]}, config=config)

        log = res["decision_log"][-1]
        self.assertEqual(log["evaluation_rating"], "error")
        self.assertEqual(log["difficulty_delta"], 0)
        self.assertIn("Evaluation failed", log["decision_reason"])

    def test_agent_max_one_retrieval_per_turn(self):
        """
        Verify that the agent cannot get stuck in an infinite retrieval loop.
        Max 1 retrieval per candidate answer turn.
        """
        class LoopMockLLM:
            def __init__(self):
                self.tools_bound = False

            def bind_tools(self, tools):
                bound_llm = LoopMockLLM()
                bound_llm.tools_bound = True
                return bound_llm

            def invoke(self, messages):
                if self.tools_bound:
                    return AIMessage(
                        content="",
                        tool_calls=[{"name": "retrieve_prep_material", "args": {"query": "loops"}, "id": "t1"}]
                    )
                else:
                    return AIMessage(content="Final Question after disabling tools.")

        checkpointer = create_memory_saver()
        agent = build_interview_agent(
            subject="Python",
            llm=LoopMockLLM(),
            checkpointer=checkpointer
        )

        config = get_session_config("test-loop-fail")
        res = agent.invoke({"messages": [HumanMessage(content="Hello!")]}, config=config)

        tool_messages = [m for m in res["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(res["messages"][-1].content, "Final Question after disabling tools.")


if __name__ == "__main__":
    unittest.main()
