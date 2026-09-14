"""
Unit and integration tests for Milestone 2: Candidate Profile & Resume Analysis.

Covers:
1. CandidateProfile schema validation
2. ProjectSummary validation
3. Mock parser extraction (testing analyze_resume with Mock LLM)
4. Empty/malformed input handling (ValueError on empty resume or role)
5. Profile injection into interviewer prompt
6. Backward compatibility with Milestone 1 prompt generation
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import ValidationError
from backend.models import CandidateProfile, ProjectSummary
from backend.parser import analyze_resume
from backend.prompts import get_interviewer_prompt


class MockStructuredLLM:
    """Mock LLM returning a predefined CandidateProfile for structured output testing."""
    def __init__(self, returned_profile: CandidateProfile):
        self.returned_profile = returned_profile
        self.last_invoked_messages = None

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.last_invoked_messages = messages
        return self.returned_profile


class TestResumeParserAndProfile(unittest.TestCase):

    def test_project_summary_validation(self):
        """Verify ProjectSummary validates required fields and sets defaults."""
        project = ProjectSummary(
            name="Placement Portal",
            technologies=["Python", "Django", "PostgreSQL"],
            description="Full-stack portal for campus placements.",
        )
        self.assertEqual(project.name, "Placement Portal")
        self.assertEqual(len(project.technologies), 3)
        self.assertIn("Django", project.technologies)

        # Missing required name or description should raise ValidationError
        with self.assertRaises(ValidationError):
            ProjectSummary(name="Incomplete Project")

    def test_candidate_profile_schema_validation(self):
        """Verify CandidateProfile enforces types, defaults, and nested schemas."""
        project = ProjectSummary(
            name="Log Aggregator",
            technologies=["FastAPI", "Redis"],
            description="Real-time log ingestion service.",
        )
        profile = CandidateProfile(
            name="Alex Chen",
            target_role="Junior Python Backend Developer",
            skills=["Python", "FastAPI", "Docker", "Redis"],
            projects=[project],
            education=["B.Tech Computer Science, 2024"],
            experience=["Python Intern at CloudScale"],
            interview_focus=["Async programming", "Redis Pub/Sub", "REST APIs"],
            potential_weak_areas=["System design scalability", "Distributed transactions"],
        )

        self.assertEqual(profile.name, "Alex Chen")
        self.assertEqual(profile.target_role, "Junior Python Backend Developer")
        self.assertEqual(len(profile.skills), 4)
        self.assertEqual(len(profile.projects), 1)
        self.assertEqual(profile.projects[0].name, "Log Aggregator")
        self.assertEqual(len(profile.interview_focus), 3)
        self.assertEqual(len(profile.potential_weak_areas), 2)

    def test_candidate_profile_defaults(self):
        """Verify defaults are properly applied when optional lists are omitted."""
        profile = CandidateProfile(target_role="Backend Developer")
        self.assertEqual(profile.name, "Candidate")
        self.assertEqual(profile.skills, [])
        self.assertEqual(profile.projects, [])
        self.assertEqual(profile.education, [])
        self.assertEqual(profile.experience, [])
        self.assertEqual(profile.interview_focus, [])
        self.assertEqual(profile.potential_weak_areas, [])

    def test_empty_and_malformed_input_handling(self):
        """Verify that analyze_resume raises ValueError on empty or whitespace inputs."""
        with self.assertRaises(ValueError) as ctx1:
            analyze_resume("", "Python Developer")
        self.assertIn("Resume text cannot be empty", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            analyze_resume("   \n\t  ", "Python Developer")
        self.assertIn("Resume text cannot be empty", str(ctx2.exception))

        with self.assertRaises(ValueError) as ctx3:
            analyze_resume("Valid resume text here...", "")
        self.assertIn("Target role cannot be empty", str(ctx3.exception))

        with self.assertRaises(ValueError) as ctx4:
            analyze_resume("Valid resume text here...", "   ")
        self.assertIn("Target role cannot be empty", str(ctx4.exception))

    def test_mock_parser_extraction(self):
        """Verify analyze_resume correctly invokes structured LLM and produces CandidateProfile."""
        expected_profile = CandidateProfile(
            name="Alex Chen",
            target_role="Junior Python Backend Developer",
            skills=["Python", "Django", "PostgreSQL"],
            projects=[
                ProjectSummary(
                    name="Placement Portal",
                    technologies=["Django", "PostgreSQL"],
                    description="Campus placement management system.",
                )
            ],
            education=["B.Tech CSE"],
            experience=["Intern at TechCorp"],
            interview_focus=["ORM optimization", "REST APIs"],
            potential_weak_areas=["Query optimization trade-offs"],
        )

        mock_llm = MockStructuredLLM(expected_profile)
        sample_resume = "Alex Chen\nPython, Django, PostgreSQL\nProject: Placement Portal"

        result = analyze_resume(
            resume_text=sample_resume,
            target_role="Junior Python Backend Developer",
            llm=mock_llm,
        )

        self.assertEqual(result.name, "Alex Chen")
        self.assertEqual(result.target_role, "Junior Python Backend Developer")
        self.assertEqual(len(result.projects), 1)
        self.assertEqual(result.projects[0].name, "Placement Portal")
        self.assertIn("Python", result.skills)

        # Verify messages passed to LLM
        self.assertIsNotNone(mock_llm.last_invoked_messages)
        self.assertEqual(len(mock_llm.last_invoked_messages), 2)
        self.assertIn("Junior Python Backend Developer", mock_llm.last_invoked_messages[0].content)

    def test_prompt_injection_with_candidate_profile(self):
        """Verify profile information is injected cleanly into the interviewer prompt."""
        profile = CandidateProfile(
            name="Alex Chen",
            target_role="Junior Python Backend Developer",
            skills=["Python", "FastAPI", "Redis"],
            projects=[
                ProjectSummary(
                    name="Log Aggregator",
                    technologies=["FastAPI", "Redis"],
                    description="Real-time log ingestion microservice.",
                )
            ],
            education=["B.Tech 2024"],
            experience=["Intern at CloudScale"],
            interview_focus=["Async endpoints", "Redis Pub/Sub"],
            potential_weak_areas=["Event loop blocking"],
        )

        prompt = get_interviewer_prompt(subject="Python", candidate_profile=profile)

        # Assert candidate-specific details are present
        self.assertIn("Alex Chen", prompt)
        self.assertIn("Junior Python Backend Developer", prompt)
        self.assertIn("Log Aggregator", prompt)
        self.assertIn("Redis", prompt)
        self.assertIn("Async endpoints", prompt)
        self.assertIn("Event loop blocking", prompt)

        # Assert instruction specifies probing rather than confirmed weakness
        self.assertIn("Treat these as topics to verify and explore", prompt)
        self.assertIn("NOT as confirmed weaknesses", prompt)

    def test_prompt_backward_compatibility_without_profile(self):
        """Verify that get_interviewer_prompt without a profile matches Milestone 1 behavior."""
        prompt = get_interviewer_prompt(subject="Python")
        self.assertIn("Target Subject: Python", prompt)
        self.assertIn("One Question at a Time", prompt)
        self.assertNotIn("Candidate Name:", prompt)
        self.assertNotIn("Personalization & Candidate Profile Guidelines:", prompt)


if __name__ == "__main__":
    unittest.main()
