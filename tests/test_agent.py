"""
Automated unit and integration tests for PlacementPrep AI Milestone 1.

Tests:
1. Prompt generation for specified subject.
2. Memory configuration and checkpointer initialization.
3. LangGraph compilation and multi-turn message retention with InMemorySaver.
4. Session isolation across different thread_id values.
5. Proper error reporting when API key is missing.
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from backend.prompts import get_interviewer_prompt
from backend.memory import create_memory_saver, get_session_config
from backend.agent import build_interview_agent, create_interviewer_llm, GEMINI_MODEL, OPENROUTER_MODEL


class MockInterviewLLM:
    """Mock LLM to test LangGraph memory and conversation loop deterministically."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0
        self.received_histories = []

    def invoke(self, messages):
        self.received_histories.append(list(messages))
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = "Thank you. That concludes this section."
        self.call_count += 1
        return AIMessage(content=resp)


class TestPlacementPrepAgent(unittest.TestCase):

    def test_interviewer_prompt_generation(self):
        """Verify the system prompt correctly includes the subject and instructions."""
        prompt = get_interviewer_prompt(subject="Python")
        self.assertIn("Python", prompt)
        self.assertIn("One Question at a Time", prompt)
        self.assertIn("Contextual Follow-up", prompt)
        self.assertIn("Adaptive Difficulty", prompt)

    def test_memory_session_config(self):
        """Verify get_session_config produces the standard LangGraph format."""
        config = get_session_config("test-session-42")
        self.assertEqual(config, {"configurable": {"thread_id": "test-session-42"}})

    def test_multiturn_conversation_and_memory_retention(self):
        """Verify that InMemorySaver preserves conversation history across multiple turns."""
        mock_responses = [
            "Welcome to the Python interview. Question 1: What is the difference between a list and a tuple?",
            "Good answer. Question 2: Since tuples are immutable, in what scenarios would you prefer them over lists?",
            "Excellent. Question 3: How does Python manage memory for integers and strings using interning?",
        ]
        mock_llm = MockInterviewLLM(mock_responses)
        checkpointer = create_memory_saver()
        agent = build_interview_agent(subject="Python", llm=mock_llm, checkpointer=checkpointer)

        session_id = "test-session-abc"
        config = get_session_config(session_id)

        # Turn 1: Initial kick-off
        r1 = agent.invoke(
            {"messages": [HumanMessage(content="Hello! Ready to begin my Python interview.")]},
            config=config,
        )
        self.assertEqual(r1["messages"][-1].content, mock_responses[0])

        # Turn 2: Candidate answers Question 1
        r2 = agent.invoke(
            {"messages": [HumanMessage(content="Lists are mutable and can change size, while tuples are immutable and fixed size.")]},
            config=config,
        )
        self.assertEqual(r2["messages"][-1].content, mock_responses[1])

        # Turn 3: Candidate answers Question 2
        r3 = agent.invoke(
            {"messages": [HumanMessage(content="I would prefer tuples for dictionary keys, record-like data that shouldn't change, and for slight memory/performance advantages.")]},
            config=config,
        )
        self.assertEqual(r3["messages"][-1].content, mock_responses[2])

        # Verify state in InMemorySaver checkpointer
        current_state = agent.get_state(config)
        messages = current_state.values["messages"]

        # 3 candidate messages + 3 interviewer responses = 6 messages total in memory
        self.assertEqual(len(messages), 6)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)
        self.assertIsInstance(messages[2], HumanMessage)
        self.assertIsInstance(messages[3], AIMessage)
        self.assertIsInstance(messages[4], HumanMessage)
        self.assertIsInstance(messages[5], AIMessage)

        # Verify that on turn 3, the mock LLM received the full accumulated history
        turn_3_received = mock_llm.received_histories[2]
        # System prompt + 4 prior messages + current user message = 6 messages to LLM
        self.assertEqual(len(turn_3_received), 6)

    def test_session_isolation_with_thread_id(self):
        """Verify that different thread_ids do not share or leak conversation history."""
        mock_responses = ["Hello session 1", "Hello session 2"]
        mock_llm = MockInterviewLLM(mock_responses)
        checkpointer = create_memory_saver()
        agent = build_interview_agent(subject="Python", llm=mock_llm, checkpointer=checkpointer)

        config_1 = get_session_config("session-1")
        config_2 = get_session_config("session-2")

        agent.invoke({"messages": [HumanMessage(content="Message for session 1")]}, config=config_1)
        agent.invoke({"messages": [HumanMessage(content="Message for session 2")]}, config=config_2)

        state_1 = agent.get_state(config_1)
        state_2 = agent.get_state(config_2)

        self.assertEqual(len(state_1.values["messages"]), 2)
        self.assertEqual(len(state_2.values["messages"]), 2)
        self.assertEqual(state_1.values["messages"][0].content, "Message for session 1")
        self.assertEqual(state_2.values["messages"][0].content, "Message for session 2")

    def test_gemini_model_configuration(self):
        """Verify the exact Gemini model is configured as gemini-3.5-flash."""
        from backend.agent import GEMINI_MODEL
        self.assertEqual(GEMINI_MODEL, "gemini-3.5-flash")

    def test_openrouter_model_configuration(self):
        """Verify the exact OpenRouter default model is configured."""
        from backend.agent import OPENROUTER_MODEL
        self.assertEqual(OPENROUTER_MODEL, "nvidia/nemotron-3-super-120b-a12b:free")

    def test_extract_message_text(self):
        """Verify text extraction handles strings, dict parts, and list formats cleanly."""
        from backend.agent import extract_message_text
        self.assertEqual(extract_message_text("Simple string"), "Simple string")
        parts = [{"type": "text", "text": "Part 1 "}, {"type": "text", "text": "Part 2"}]
        self.assertEqual(extract_message_text(parts), "Part 1 Part 2")

    def test_missing_api_key_raises_error(self):
        """Verify that missing API key raises ValueError for specified or active provider."""
        with self.assertRaises(ValueError) as ctx:
            create_interviewer_llm(provider="gemini", api_key="")
        self.assertIn("GEMINI_API_KEY not found", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            create_interviewer_llm(provider="openrouter", api_key="")
        self.assertIn("OPENROUTER_API_KEY not found", str(ctx.exception))

    def test_openrouter_factory_configuration(self):
        """Verify OpenRouter factory returns properly configured ChatOpenAI instance."""
        from langchain_openai import ChatOpenAI
        llm = create_interviewer_llm(provider="openrouter", api_key="sk-or-test-key")
        self.assertIsInstance(llm, ChatOpenAI)
        self.assertEqual(llm.model_name, "nvidia/nemotron-3-super-120b-a12b:free")
        self.assertEqual(str(llm.openai_api_base).rstrip("/"), "https://openrouter.ai/api/v1")
        self.assertEqual(llm.temperature, 0.7)

    def test_gemini_factory_configuration(self):
        """Verify Gemini factory returns properly configured ChatGoogleGenerativeAI instance."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = create_interviewer_llm(provider="gemini", api_key="dummy-gemini-key", model_name="gemini-3.5-flash")
        self.assertIsInstance(llm, ChatGoogleGenerativeAI)
        self.assertEqual(llm.model, "gemini-3.5-flash")
        self.assertEqual(llm.temperature, 0.7)

    def test_provider_selection_env_override(self):
        """Verify LLM_PROVIDER environment variable controls provider selection."""
        from unittest.mock import patch
        from langchain_openai import ChatOpenAI
        from langchain_google_genai import ChatGoogleGenerativeAI

        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or-test"}):
            llm = create_interviewer_llm()
            self.assertIsInstance(llm, ChatOpenAI)

        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "dummy-gemini"}):
            llm = create_interviewer_llm()
            self.assertIsInstance(llm, ChatGoogleGenerativeAI)

        with patch.dict(os.environ, {"LLM_PROVIDER": "unsupported_provider"}):
            with self.assertRaises(ValueError) as ctx:
                create_interviewer_llm()
            self.assertIn("Invalid LLM_PROVIDER", str(ctx.exception))

    def test_provider_selection_fallback(self):
        """Verify fallback logic when LLM_PROVIDER is absent."""
        from unittest.mock import patch
        from langchain_openai import ChatOpenAI
        from langchain_google_genai import ChatGoogleGenerativeAI

        # When OPENROUTER_API_KEY exists -> OpenRouter
        with patch.dict(os.environ, {"LLM_PROVIDER": "", "OPENROUTER_API_KEY": "sk-or-test", "GEMINI_API_KEY": "dummy"}):
            llm = create_interviewer_llm()
            self.assertIsInstance(llm, ChatOpenAI)

        # When OPENROUTER_API_KEY is empty/absent -> Gemini
        with patch.dict(os.environ, {"LLM_PROVIDER": "", "OPENROUTER_API_KEY": "", "GEMINI_API_KEY": "dummy"}):
            llm = create_interviewer_llm()
            self.assertIsInstance(llm, ChatGoogleGenerativeAI)

    def test_structured_output_construction(self):
        """Verify structured output runnable can be constructed for both providers without live API calls."""
        from backend.models import CandidateProfile
        openrouter_llm = create_interviewer_llm(provider="openrouter", api_key="sk-or-test")
        gemini_llm = create_interviewer_llm(provider="gemini", api_key="dummy-gemini")

        or_structured = openrouter_llm.with_structured_output(CandidateProfile, method="function_calling")
        self.assertIsNotNone(or_structured)

        gemini_structured = gemini_llm.with_structured_output(CandidateProfile)
        self.assertIsNotNone(gemini_structured)


if __name__ == "__main__":
    unittest.main()
