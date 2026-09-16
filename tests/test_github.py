import pytest
from unittest.mock import patch, MagicMock
from backend.github.client import GitHubClient
from backend.github.analyzer import select_relevant_repositories, analyze_repository, fetch_and_analyze_github_profile
from backend.models import ProjectProfile

@pytest.fixture
def github_client():
    return GitHubClient(token="fake_token")

def test_github_client_get_user_profile(github_client):
    with patch("backend.github.client.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"login": "testuser"}
        mock_get.return_value = mock_response
        
        profile = github_client.get_user_profile("testuser")
        assert profile["login"] == "testuser"

def test_github_client_error_handling(github_client):
    with patch("backend.github.client.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Rate limit exceeded"
        mock_get.return_value = mock_response
        
        profile = github_client.get_user_profile("testuser")
        assert profile is None

def test_select_relevant_repositories():
    repos = [
        {"name": "forked_repo", "fork": True, "stargazers_count": 10},
        {"name": "repo1", "fork": False, "stargazers_count": 5},
        {"name": "repo2", "fork": False, "stargazers_count": 15},
    ]
    selected = select_relevant_repositories(repos, limit=2)
    assert len(selected) == 2
    assert selected[0]["name"] == "repo2"
    assert selected[1]["name"] == "repo1"

def test_analyze_repository(github_client):
    repo_data = {"name": "testrepo", "description": "test desc"}
    
    with patch.object(github_client, "get_repository_readme", return_value="Test README"), \
         patch.object(github_client, "get_repository_tree", return_value=["main.py"]):
         
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        
        mock_profile = ProjectProfile(
            project_name="testrepo",
            description="Test desc",
            technologies=["Python"],
        )
        mock_structured_llm.invoke.return_value = mock_profile
        
        profile = analyze_repository(github_client, "testuser", repo_data, llm=mock_llm)
        
        assert profile is not None
        assert profile.project_name == "testrepo"
        assert "Python" in profile.technologies

def test_fetch_and_analyze_empty_repos():
    with patch("backend.github.client.GitHubClient.get_user_profile", return_value={"login": "testuser"}), \
         patch("backend.github.client.GitHubClient.get_user_repositories", return_value=[]):
        
        profile = fetch_and_analyze_github_profile("testuser", llm=MagicMock())
        assert profile is not None
        assert profile.username == "testuser"
        assert len(profile.projects) == 0
