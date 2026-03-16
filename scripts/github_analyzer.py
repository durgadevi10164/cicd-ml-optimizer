#!/usr/bin/env python3
import requests
import os
from dotenv import load_dotenv
import re

load_dotenv()

class GitHubRepoAnalyzer:
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        self.headers = {'Authorization': f'token {self.token}'} if self.token else {}
    
    def parse_github_url(self, url):
        """Extract owner and repo from GitHub URL"""
        # Supports: https://github.com/owner/repo or github.com/owner/repo
        pattern = r'github\.com[/:]([^/]+)/([^/\.]+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2)
        raise ValueError("Invalid GitHub URL")
    
    def get_repo_info(self, url):
        """Get comprehensive repo information"""
        owner, repo = self.parse_github_url(url)
        
        # Main repo data
        repo_url = f'https://api.github.com/repos/{owner}/{repo}'
        response = requests.get(repo_url, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch repo: {response.json().get('message', 'Unknown error')}")
        
        data = response.json()
        
        # Get languages
        languages_url = data['languages_url']
        languages = requests.get(languages_url, headers=self.headers).json()
        
        # Count dependencies
        dependencies = self._count_dependencies(owner, repo)
        
        # Get file count (approximate from tree)
        file_count = self._estimate_file_count(owner, repo, data.get('default_branch', 'main'))
        
        return {
            'repo_name': data['full_name'],
            'repo_size_mb': data['size'] / 1024,  # GitHub gives size in KB
            'num_files': file_count,
            'num_dependencies': dependencies,
            'languages': list(languages.keys()),
            'primary_language': data.get('language', 'Unknown'),
            'stars': data['stargazers_count'],
            'forks': data['forks_count'],
            'open_issues': data['open_issues_count'],
            'default_branch': data.get('default_branch', 'main'),
            'created_at': data['created_at'],
            'updated_at': data['updated_at']
        }
    
    def _count_dependencies(self, owner, repo):
        """Count dependencies from package files"""
        dependency_count = 0
        
        # Check for different package managers
        package_files = {
            'package.json': self._count_npm_deps,
            'requirements.txt': self._count_python_deps,
            'pom.xml': self._count_maven_deps,
            'build.gradle': self._count_gradle_deps,
            'Gemfile': self._count_ruby_deps,
            'go.mod': self._count_go_deps,
        }
        
        for filename, counter_func in package_files.items():
            try:
                file_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{filename}'
                response = requests.get(file_url, headers=self.headers)
                
                if response.status_code == 200:
                    content = response.json()
                    if content.get('download_url'):
                        file_content = requests.get(content['download_url']).text
                        dependency_count += counter_func(file_content)
                        break  # Use first package file found
            except:
                continue
        
        # Default estimate if no package file found
        return dependency_count if dependency_count > 0 else 100
    
    def _count_npm_deps(self, content):
        """Count NPM dependencies"""
        try:
            import json
            data = json.loads(content)
            deps = len(data.get('dependencies', {}))
            dev_deps = len(data.get('devDependencies', {}))
            return deps + dev_deps
        except:
            return 50
    
    def _count_python_deps(self, content):
        """Count Python dependencies"""
        lines = [l.strip() for l in content.split('\n') if l.strip() and not l.startswith('#')]
        return len(lines)
    
    def _count_maven_deps(self, content):
        """Count Maven dependencies"""
        return content.count('<dependency>')
    
    def _count_gradle_deps(self, content):
        """Count Gradle dependencies"""
        return content.count('implementation') + content.count('compile')
    
    def _count_ruby_deps(self, content):
        """Count Ruby gems"""
        return content.count('gem ')
    
    def _count_go_deps(self, content):
        """Count Go dependencies"""
        return content.count('require')
    
    def _estimate_file_count(self, owner, repo, branch):
        """Estimate file count from repo tree"""
        try:
            tree_url = f'https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1'
            response = requests.get(tree_url, headers=self.headers)
            
            if response.status_code == 200:
                tree = response.json().get('tree', [])
                # Count only files (type='blob'), exclude directories
                files = [item for item in tree if item['type'] == 'blob']
                return len(files)
        except:
            pass
        
        # Fallback estimate based on size
        return 500  # Conservative estimate
    
    def prepare_for_ml(self, repo_info):
        """Convert repo info to ML model input format"""
        # Estimate test count based on file count (rough heuristic)
        test_count = int(repo_info['num_files'] * 0.15)  # ~15% of files are tests
        
        return {
            'repo_size_mb': int(repo_info['repo_size_mb']),
            'num_files': repo_info['num_files'],
            'num_dependencies': repo_info['num_dependencies'],
            'test_count': test_count
        }

# Test it
if __name__ == "__main__":
    analyzer = GitHubRepoAnalyzer()
    
    # Test with a popular repo
    test_urls = [
        "https://github.com/facebook/react",
        "https://github.com/microsoft/vscode",
        "https://github.com/django/django"
    ]
    
    print("Testing GitHub Analyzer...")
    print("=" * 70)
    
    for url in test_urls[:1]:  # Test with just one to avoid rate limits
        try:
            print(f"\nAnalyzing: {url}")
            info = analyzer.get_repo_info(url)
            
            print(f"\n📊 Repository Information:")
            print(f"   Name: {info['repo_name']}")
            print(f"   Size: {info['repo_size_mb']:.2f} MB")
            print(f"   Files: {info['num_files']:,}")
            print(f"   Dependencies: {info['num_dependencies']}")
            print(f"   Language: {info['primary_language']}")
            print(f"   Stars: {info['stars']:,}")
            
            print(f"\n🤖 ML Model Input:")
            ml_input = analyzer.prepare_for_ml(info)
            for key, value in ml_input.items():
                print(f"   {key}: {value}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

