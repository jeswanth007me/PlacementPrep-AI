import os
import requests
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub client. 
        Uses provided token or falls back to GITHUB_TOKEN environment variable.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
        self.base_url = "https://api.github.com"
        
    def _handle_response(self, response: requests.Response) -> Optional[Any]:
        if response.status_code == 200:
            return response.json()
        elif response.status_code in [401, 403, 404]:
            logger.warning(f"GitHub API Error {response.status_code}: {response.text}")
            return None
        else:
            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                logger.warning(f"GitHub API HTTP Error: {e}")
                return None
            
    def get_user_profile(self, username: Optional[str] = None) -> Optional[Dict]:
        """Fetch user profile. If no username is provided, fetches the authenticated user."""
        url = f"{self.base_url}/users/{username}" if username else f"{self.base_url}/user"
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)
        
    def get_user_repositories(self, username: Optional[str] = None, sort="updated", per_page=100) -> List[Dict]:
        """Fetch user repositories. Excludes forks by default in the caller."""
        url = f"{self.base_url}/users/{username}/repos" if username else f"{self.base_url}/user/repos"
        params = {"sort": sort, "per_page": per_page}
        response = requests.get(url, headers=self.headers, params=params)
        repos = self._handle_response(response)
        return repos if isinstance(repos, list) else []
        
    def get_repository_readme(self, owner: str, repo: str) -> Optional[str]:
        """Fetch the raw README content for a repository."""
        url = f"{self.base_url}/repos/{owner}/{repo}/readme"
        headers = self.headers.copy()
        headers["Accept"] = "application/vnd.github.v3.raw"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        return None
        
    def get_repository_tree(self, owner: str, repo: str, default_branch: str) -> List[str]:
        """Fetch the top-level repository tree."""
        url = f"{self.base_url}/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
        response = requests.get(url, headers=self.headers)
        data = self._handle_response(response)
        if data and "tree" in data:
            # Return a simplified list of files, limit to 100 paths
            files = [item["path"] for item in data["tree"] if item.get("type") == "blob"]
            return files[:100]
        return []
