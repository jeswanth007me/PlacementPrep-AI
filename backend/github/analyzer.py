import logging
from typing import List, Optional, Dict
from langchain_core.messages import HumanMessage, SystemMessage

from backend.models import ProjectProfile, GitHubProfile
from backend.github.client import GitHubClient

logger = logging.getLogger(__name__)

PROJECT_ANALYZER_SYSTEM_PROMPT = """You are an expert technical recruiter and software architect for PlacementPrep AI.
Your task is to analyze a candidate's GitHub repository and extract a structured Project Profile.

Repository Information:
Name: {repo_name}
Description: {repo_description}
Language: {repo_language}
Topics: {repo_topics}
File Tree (partial):
{file_tree}

README Content (truncated if too long):
{readme_content}

Instructions:
1. Extract the project's primary technologies, frameworks, and features.
2. Summarize the likely architecture based on the README and file tree.
3. Identify important files (e.g., main.py, package.json, Dockerfile) present in the tree.
4. Generate 2-3 potential technical interview topics or concepts the interviewer could ask about this project.
5. If the README is empty or lacks detail, infer as much as safely possible from the description and file tree, but DO NOT hallucinate features that aren't evident.
"""

def select_relevant_repositories(repos: List[Dict], target_role: str = "", limit: int = 3) -> List[Dict]:
    """
    Select the most relevant non-fork repositories based on heuristics.
    """
    valid_repos = [repo for repo in repos if not repo.get("fork")]
    
    # Sort by stars and recently updated
    valid_repos.sort(key=lambda x: (x.get("stargazers_count", 0), x.get("updated_at", "")), reverse=True)
    
    # Basic relevance (optional, can be improved)
    # For now, just take the top ones.
    return valid_repos[:limit]

def analyze_repository(
    client: GitHubClient,
    owner: str,
    repo_data: Dict,
    llm: Optional[object] = None,
) -> Optional[ProjectProfile]:
    """
    Analyze a repository using LLM to generate a ProjectProfile.
    """
    repo_name = repo_data.get("name", "")
    if not repo_name:
        return None
        
    readme = client.get_repository_readme(owner, repo_name) or "No README provided."
    default_branch = repo_data.get("default_branch", "main")
    file_tree = client.get_repository_tree(owner, repo_name, default_branch)
    
    if llm is None:
        from backend.agent import create_interviewer_llm
        llm = create_interviewer_llm()
        
    structured_llm = llm.with_structured_output(ProjectProfile)
    
    # Truncate README to avoid context overflow if it's massive
    truncated_readme = readme[:10000] 
    tree_str = "\n".join(file_tree[:50])
    
    system_prompt = PROJECT_ANALYZER_SYSTEM_PROMPT.format(
        repo_name=repo_name,
        repo_description=repo_data.get("description", "No description"),
        repo_language=repo_data.get("language", "Unknown"),
        repo_topics=", ".join(repo_data.get("topics", [])),
        file_tree=tree_str,
        readme_content=truncated_readme
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please analyze this repository and return the ProjectProfile.")
    ]
    
    try:
        profile = structured_llm.invoke(messages)
        # Ensure project name is set to repo name
        if not profile.project_name:
            profile.project_name = repo_name
        return profile
    except Exception as e:
        logger.error(f"Failed to analyze repository {repo_name}: {e}")
        return None

def fetch_and_analyze_github_profile(
    username: str, 
    target_role: str = "",
    token: Optional[str] = None,
    llm: Optional[object] = None,
) -> Optional[GitHubProfile]:
    """
    Main entry point to fetch a candidate's GitHub profile and analyze top repositories.
    """
    client = GitHubClient(token=token)
    user_data = client.get_user_profile(username)
    
    if not user_data:
        logger.warning(f"Could not fetch profile for GitHub user: {username}")
        return None
        
    repos_data = client.get_user_repositories(username)
    # We limit to 2 for speed during interview setup/testing.
    selected_repos = select_relevant_repositories(repos_data, target_role, limit=2)
    
    project_profiles = []
    for repo in selected_repos:
        profile = analyze_repository(client, username, repo, llm=llm)
        if profile:
            project_profiles.append(profile)
            
    return GitHubProfile(
        username=user_data.get("login", username),
        profile_url=user_data.get("html_url", ""),
        projects=project_profiles
    )
