from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.optimizer import PipelineOptimizer
from scripts.github_analyzer import GitHubRepoAnalyzer

app = Flask(__name__)
CORS(app)

optimizer = PipelineOptimizer()
github_analyzer = GitHubRepoAnalyzer()

# Store last analysis for chatbot context
last_analysis = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze-github', methods=['POST'])
def analyze_github():
    global last_analysis
    try:
        data = request.json
        github_url = data.get('github_url')
        
        if not github_url:
            return jsonify({'success': False, 'error': 'GitHub URL required'}), 400
        
        repo_info = github_analyzer.get_repo_info(github_url)
        ml_input = github_analyzer.prepare_for_ml(repo_info)
        recommendation = optimizer.optimize_resources(
            repo_size=ml_input['repo_size_mb'],
            num_files=ml_input['num_files'],
            num_deps=ml_input['num_dependencies'],
            test_count=ml_input['test_count']
        )
        
        # Store for chatbot context
        last_analysis = {
            'repo_info': repo_info,
            'ml_input': ml_input,
            'recommendation': recommendation['recommended']
        }
        
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
    global last_analysis
    try:
        data = request.json
        result = optimizer.optimize_resources(
            repo_size=int(data['repo_size']),
            num_files=int(data['num_files']),
            num_deps=int(data['num_deps']),
            test_count=int(data['test_count'])
        )
        
        # Store for chatbot
        last_analysis = {
            'ml_input': {
                'repo_size_mb': data['repo_size'],
                'num_files': data['num_files'],
                'num_dependencies': data['num_deps'],
                'test_count': data['test_count']
            },
            'recommendation': result['recommended']
        }
        
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    """AI chatbot endpoint"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not last_analysis:
            return jsonify({
                'success': True,
                'response': "Please analyze a repository first, then I can answer your questions about the recommendations!"
            })
        
        # Generate intelligent response based on context
        response = generate_chatbot_response(user_message, last_analysis)
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
def generate_chatbot_response(user_message, context):
    """Generate context-aware responses"""
    
    msg = user_message.lower()
    rec = context['recommendation']
    ml_input = context['ml_input']
    
    # Pattern matching for common questions
    if 'why' in msg and ('high' in msg or 'long' in msg or 'slow' in msg):
        return f"""Your build time is estimated at **{rec['predicted_duration_min']:.1f} minutes** because:

1. **Repository size**: {ml_input['repo_size_mb']} MB - Larger repos take longer to clone and process
2. **Number of files**: {ml_input['num_files']:,} files - More files = more compilation/bundling
3. **Dependencies**: {ml_input['num_dependencies']} packages - Each dependency adds download & install time
4. **Tests**: ~{ml_input['test_count']} tests to run

💡 **Tip**: The recommended {rec['cpu_cores']} CPU cores will help parallelize the build and reduce time by ~30%."""

    elif 'why' in msg and ('cpu' in msg or 'cores' in msg):
        return f"""I recommended **{rec['cpu_cores']} CPU cores** because:

- Your project has **{ml_input['num_files']:,} files** that can be compiled in parallel
- With {ml_input['num_dependencies']} dependencies, parallel processing speeds up installation
- This configuration gives you the best **cost/performance balance**
- Success probability: **{rec['success_probability']:.1f}%**

📊 Using fewer cores would save money but increase build time by 40-60%."""

    elif 'why' in msg and ('memory' in msg or 'ram' in msg):
        return f"""I recommended **{rec['memory_gb']} GB RAM** because:

- Your repository size ({ml_input['repo_size_mb']} MB) needs adequate memory for caching
- Prevents memory swapping during parallel compilation
- Ensures {ml_input['test_count']} tests can run without OOM errors
- Optimizes for **{rec['success_probability']:.1f}% success rate**

⚠️ Less RAM could cause build failures due to memory exhaustion."""

    elif 'cost' in msg and ('reduce' in msg or 'save' in msg or 'lower' in msg):
        current_cost = rec['estimated_cost_usd']
        lower_cost = current_cost * 0.6
        return f"""To reduce costs:

**Current**: ${current_cost:.3f} per build with {rec['cpu_cores']} cores, {rec['memory_gb']}GB RAM

**Option 1**: Use 2 CPU cores + 4GB RAM
- Cost: ~${lower_cost:.3f} (40% savings)
- Build time: +8-10 minutes
- Success rate: ~70% (vs current {rec['success_probability']:.1f}%)

**Option 2**: Run builds during off-peak hours
- Use spot instances (60% cheaper)
- Same performance, lower cost

💰 For 100 builds/month: Save ~${(current_cost - lower_cost) * 100:.2f}/month"""

    elif 'success' in msg or 'fail' in msg:
        return f"""Your build has a **{rec['success_probability']:.1f}% predicted success rate** with the recommended configuration.

**Factors affecting success**:
✅ Adequate resources ({rec['cpu_cores']} cores, {rec['memory_gb']}GB RAM)
✅ Sufficient memory for {ml_input['num_files']:,} files
✅ Proper CPU allocation for {ml_input['num_dependencies']} dependencies

**To improve success rate**:
1. Increase RAM to handle memory-intensive tests
2. Add more CPU cores for parallel execution
3. Optimize your test suite to run faster"""

    elif 'how' in msg and 'faster' in msg:
        faster_cpu = min(rec['cpu_cores'] * 2, 16)
        time_saved = rec['predicted_duration_min'] * 0.35
        return f"""To make builds **faster**:

**Option 1**: Increase CPU cores
- Use {faster_cpu} cores instead of {rec['cpu_cores']}
- Save ~{time_saved:.1f} minutes per build
- Cost increase: ~$0.15-0.25

**Option 2**: Optimize your build
- Enable build caching
- Use incremental builds
- Parallelize tests
- Remove unused dependencies

**Option 3**: Use faster build machines
- SSD storage (2x faster I/O)
- More RAM for caching"""

    elif 'compare' in msg or 'alternative' in msg or 'option' in msg:
        return f"""Here are alternative configurations:

**Current Recommendation**:
- {rec['cpu_cores']} cores, {rec['memory_gb']}GB RAM
- Time: {rec['predicted_duration_min']:.1f} min
- Cost: ${rec['estimated_cost_usd']:.3f}
- Success: {rec['success_probability']:.1f}%

**Budget Option**:
- 2 cores, 4GB RAM
- Time: ~{rec['predicted_duration_min'] * 1.5:.1f} min
- Cost: ~${rec['estimated_cost_usd'] * 0.4:.3f}
- Success: ~70%

**Performance Option**:
- 16 cores, 32GB RAM  
- Time: ~{rec['predicted_duration_min'] * 0.6:.1f} min
- Cost: ~${rec['estimated_cost_usd'] * 2.2:.3f}
- Success: ~98%

Which matters more to you: speed, cost, or reliability?"""

    elif 'less' in msg and ('ram' in msg or 'memory' in msg):
        current_ram = rec['memory_gb']
        less_ram = max(current_ram // 2, 4)
        time_increase = rec['predicted_duration_min'] * 0.15
        return f"""If you use **{less_ram}GB RAM** instead of {current_ram}GB:

**Risks:**
⚠️ **Out of Memory errors** - Your {ml_input['num_files']:,} files need adequate RAM
⚠️ **Success rate drops** to ~75% (vs current {rec['success_probability']:.1f}%)
⚠️ **Build may fail** randomly during peak memory usage
⚠️ **Slower builds** - May add ~{time_increase:.1f} minutes due to swapping

**Benefits:**
💰 Save ~$0.08-0.12 per build
💰 Monthly savings: ~$8-12 (for 100 builds)

**My Recommendation:** 
Stick with {current_ram}GB RAM. The cost savings aren't worth the reliability risk for your project size."""

    elif 'affect' in msg and 'success' in msg:
        return f"""**Success rate ({rec['success_probability']:.1f}%) depends on:**

**1. Memory Allocation** ⭐⭐⭐ (Most Important)
- Current: {rec['memory_gb']}GB RAM
- Your repo size ({ml_input['repo_size_mb']}MB) + {ml_input['num_files']:,} files need this
- **Too little RAM** = Out of Memory crashes
- **Adequate RAM** = Stable builds

**2. CPU Resources** ⭐⭐
- Current: {rec['cpu_cores']} cores
- Affects parallel compilation and test execution
- **Too few cores** = Timeouts on large projects
- **Right cores** = Smooth parallel processing

**3. Repository Complexity** ⭐⭐
- Files: {ml_input['num_files']:,}
- Dependencies: {ml_input['num_dependencies']}
- Tests: ~{ml_input['test_count']}
- **More complex** = needs more resources for stability

**4. Build Process Type** ⭐
- Compilation-heavy projects need more CPU
- Test-heavy projects need more memory
- Your project falls in the balanced category

**Bottom Line:** Your recommended config ({rec['cpu_cores']} cores, {rec['memory_gb']}GB) gives you {rec['success_probability']:.1f}% success - a solid, reliable setup!"""

    elif ('4' in msg or 'four' in msg) and ('core' in msg or 'cpu' in msg):
        faster_time = rec['predicted_duration_min'] * 0.65
        slower_time = rec['predicted_duration_min'] * 1.4
        cheaper_cost = rec['estimated_cost_usd'] * 0.6
        return f"""**Configuration Comparison:**

**4 CPU Cores (Budget Option)**
- Build Time: ~{slower_time:.1f} min (+40%)
- Cost: ${cheaper_cost:.3f} (-40%)
- Success Rate: ~80%
- Best For: Non-critical builds

**{rec['cpu_cores']} CPU Cores (Recommended)** ✅
- Build Time: {rec['predicted_duration_min']:.1f} min
- Cost: ${rec['estimated_cost_usd']:.3f}
- Success Rate: {rec['success_probability']:.1f}%
- Best For: Your project size

**16 CPU Cores (Performance)**
- Build Time: ~{faster_time:.1f} min (-35%)
- Cost: ${rec['estimated_cost_usd'] * 1.8:.3f} (+80%)
- Success Rate: ~96%
- Best For: Time-critical deployments

**Which matters more:** ⚡ Speed or 💰 Cost?"""

    elif 'help' in msg or 'what can you' in msg or 'questions' in msg:
        return """I can answer questions like:

❓ **Performance**:
- "Why is my build time so high?"
- "How can I make builds faster?"

💰 **Cost**:
- "How can I reduce costs?"
- "What's the cheapest option?"

⚙️ **Configuration**:
- "Why did you recommend 8 CPU cores?"
- "Why do I need 16GB RAM?"
- "Show me alternative options"

📊 **Success Rate**:
- "Why is my success rate low?"
- "How can I improve reliability?"

Just ask me anything about your build configuration!"""

    else:
        # Default response
        return f"""Based on your project ({ml_input['num_files']:,} files, {ml_input['repo_size_mb']}MB):

**Recommended**: {rec['cpu_cores']} cores, {rec['memory_gb']}GB RAM
**Build Time**: ~{rec['predicted_duration_min']:.1f} minutes  
**Cost**: ${rec['estimated_cost_usd']:.3f} per build
**Success Rate**: {rec['success_probability']:.1f}%

Try asking:
- "Why is my build time high?"
- "How can I reduce costs?"
- "Why {rec['cpu_cores']} CPU cores?"
- "Show me alternatives"

What would you like to know?"""

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 CI/CD Pipeline Optimizer - Dashboard with AI Chatbot")
    print("=" * 70)
    print("📊 Access at: http://localhost:5000")
    print("💬 Ask the AI about your build configurations!")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
