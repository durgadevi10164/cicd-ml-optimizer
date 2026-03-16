from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.optimizer import PipelineOptimizer
from scripts.github_analyzer import GitHubRepoAnalyzer

app = Flask(__name__)
CORS(app)

optimizer = PipelineOptimizer()
github_analyzer = GitHubRepoAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze-github', methods=['POST'])
def analyze_github():
    """Analyze GitHub repo and get optimization"""
    try:
        data = request.json
        github_url = data.get('github_url')
        
        if not github_url:
            return jsonify({'success': False, 'error': 'GitHub URL required'}), 400
        
        # Step 1: Analyze GitHub repo
        repo_info = github_analyzer.get_repo_info(github_url)
        
        # Step 2: Prepare for ML
        ml_input = github_analyzer.prepare_for_ml(repo_info)
        
        # Step 3: Get optimization recommendation
        recommendation = optimizer.optimize_resources(
            repo_size=ml_input['repo_size_mb'],
            num_files=ml_input['num_files'],
            num_deps=ml_input['num_dependencies'],
            test_count=ml_input['test_count']
        )
        
        return jsonify({
            'success': True,
            'repo_info': repo_info,
            'ml_input': ml_input,
            'recommendation': recommendation
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/optimize-manual', methods=['POST'])
def optimize_manual():
    """Manual input optimization"""
    try:
        data = request.json
        result = optimizer.optimize_resources(
            repo_size=int(data['repo_size']),
            num_files=int(data['num_files']),
            num_deps=int(data['num_deps']),
            test_count=int(data['test_count'])
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 CI/CD Pipeline Optimizer - Dashboard")
    print("=" * 70)
    print("📊 Access at: http://localhost:5000")
    print("🔗 Try pasting a GitHub URL to analyze!")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
