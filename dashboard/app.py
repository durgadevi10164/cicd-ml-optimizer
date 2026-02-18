#!/usr/bin/env python3
"""
CI/CD ML Optimizer Dashboard
Flask web application for build optimization with AI Chat
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.predict import predict_build_time
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# Load environment variables and setup Gemini
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Load model metadata
with open('models/metadata.json') as f:
    metadata = json.load(f)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', metadata=metadata)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict build time for given configuration"""
    try:
        config = request.json
        predicted_time = predict_build_time(config)
        
        return jsonify({
            'success': True,
            'predicted_time': round(predicted_time, 2)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """Find optimal resource configuration"""
    try:
        config = request.json
        
        # Get optimization results
        cpu_options = [2, 4, 8, 16]
        mem_options = [4, 8, 16, 32]
        
        results = []
        for cpu in cpu_options:
            for mem in mem_options:
                test_config = config.copy()
                test_config['cpu_cores'] = cpu
                test_config['memory_gb'] = mem
                
                pred_time = predict_build_time(test_config)
                
                cpu_cost_hr = {2: 0.05, 4: 0.10, 8: 0.20, 16: 0.40}[cpu]
                mem_cost_hr = {4: 0.02, 8: 0.04, 16: 0.08, 32: 0.16}[mem]
                cost = (cpu_cost_hr + mem_cost_hr) * (pred_time / 60)
                
                results.append({
                    'cpu_cores': cpu,
                    'memory_gb': mem,
                    'predicted_time': round(pred_time, 2),
                    'cost': round(cost, 4)
                })
        
        # Sort by cost
        results.sort(key=lambda x: x['cost'])
        
        return jsonify({
            'success': True,
            'recommendations': results[:5],
            'all_configs': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/stats')
def api_stats():
    """Get dataset statistics"""
    try:
        df = pd.read_csv('data/builds.csv')
        
        stats = {
            'total_builds': len(df),
            'success_rate': float((df['status'] == 'success').mean()),
            'avg_build_time': float(df['build_time_minutes'].mean()),
            'avg_cost': float(df['cost_usd'].mean()),
            'total_cost': float(df['cost_usd'].sum()),
            'by_branch': df.groupby('branch_type').agg({                'build_time_minutes': 'mean',
                'cost_usd': 'mean'
            }).round(2).to_dict()
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """Chat with AI about build optimization - MOCK VERSION (No API needed)"""
    try:
        data = request.json
        user_message = data.get('message', '').lower()
        build_context = data.get('context', {})
        
        # Smart rule-based responses (works without any API!)
        if 'why' in user_message and ('long' in user_message or 'slow' in user_message or 'time' in user_message):
            response = f"Your build takes {build_context.get('predicted_time', 'N/A')} minutes due to several factors:\n\n1. **Large repository**: {build_context.get('repo_size_mb', 'N/A')}MB with {build_context.get('num_files', 'N/A')} files\n2. **Test suite**: {build_context.get('test_count', 'N/A')} tests to run\n3. **Cache efficiency**: Only {build_context.get('cache_hit_rate', 'N/A')} cache hit rate means many dependencies are re-downloaded\n4. **Dependencies**: {build_context.get('dependencies', 'N/A')} packages to install\n\nTo speed it up: improve caching, run tests in parallel, and use the recommended {build_context.get('recommended_cpu', 'N/A')} cores configuration."
        
        elif 'cost' in user_message and 'high' in user_message:
            response = f"The cost is ${build_context.get('estimated_cost', 'N/A')} because you're using {build_context.get('cpu_cores', 'N/A')} CPU cores and {build_context.get('memory_gb', 'N/A')}GB RAM - which is over-provisioned for your needs.\n\nYour {build_context.get('branch_type', 'N/A')} branch doesn't need that much power! By switching to our recommended {build_context.get('recommended_cpu', 'N/A')} cores and {build_context.get('recommended_memory', 'N/A')}GB RAM, you'll save approximately {build_context.get('savings_percent', 'N/A')}% per build while maintaining good performance."
        
        elif 'reduce' in user_message or 'improve' in user_message or 'faster' in user_message or 'optimize' in user_message:
            response = f"Here are the top 4 ways to reduce your {build_context.get('predicted_time', 'N/A')} minute build time:\n\n1. **Boost cache hit rate** from {build_context.get('cache_hit_rate', 'N/A')} to 0.8+ by caching dependencies properly\n2. **Right-size resources**: Use {build_context.get('recommended_cpu', 'N/A')} cores and {build_context.get('recommended_memory', 'N/A')}GB RAM (optimal balance)\n3. **Parallel testing**: Run your {build_context.get('test_count', 'N/A')} tests in parallel instead of sequentially\n4. **Split test suites**: Break large test suites into smaller, faster chunks\n\nThese changes could cut your build time in half!"
        
        elif '2 cores' in user_message or '4 cores' in user_message or 'recommend' in user_message:
            response = f"I recommended **{build_context.get('recommended_cpu', 'N/A')} cores and {build_context.get('recommended_memory', 'N/A')}GB RAM** because:\n\n✅ **Cost-efficient**: ${build_context.get('estimated_cost', 'N/A')} per build (saves {build_context.get('savings_percent', 'N/A')}%)\n✅ **Right-sized**: Perfect for {build_context.get('branch_type', 'N/A')} branch with {build_context.get('repo_size_mb', 'N/A')}MB repo\n✅ **Build time**: Completes in ~{build_context.get('predicted_time', 'N/A')} minutes\n\nHigher specs won't make your build much faster, but they'll cost significantly more. This is the sweet spot!"
        
        elif 'cache' in user_message:
            response = f"Your current cache hit rate is **{build_context.get('cache_hit_rate', 'N/A')}**, which means you're re-downloading dependencies too often.\n\n**How to improve caching:**\n1. Cache dependency directories (node_modules, pip packages)\n2. Use lock files (package-lock.json, requirements.txt)\n3. Cache build artifacts between stages\n4. Enable incremental builds\n\nGood caching (0.8+) can reduce build times by 30-50%!"
        
        elif 'test' in user_message:
            response = f"With **{build_context.get('test_count', 'N/A')} tests**, parallelization is crucial! Split tests into groups and run concurrently. This can reduce test time from hours to minutes!"
        
        else:
            response = f"I've analyzed your **{build_context.get('repo_size_mb', 'N/A')}MB repository** with {build_context.get('test_count', 'N/A')} tests.\n\n**My recommendation:** {build_context.get('recommended_cpu', 'N/A')} cores and {build_context.get('recommended_memory', 'N/A')}GB RAM\n**Build time:** ~{build_context.get('predicted_time', 'N/A')} minutes\n**Cost:** ${build_context.get('estimated_cost', 'N/A')}\n**Savings:** {build_context.get('savings_percent', 'N/A')}%\n\nAsk me: Why is my build time so long? How can I reduce costs?"
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
       
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 CI/CD ML Optimizer Dashboard Starting...")
    print("="*60)
    print(f"\n📊 Model Info:")
    print(f"   Test R²: {metadata['metrics']['test_r2']}")
    print(f"   Test MAE: {metadata['metrics']['test_mae']} minutes")
    print(f"\n🌐 Dashboard URL: http://localhost:5000")
    print(f"\n🤖 AI Chat: Enabled (Smart Rule-Based Assistant)")
    print("\n⚡ Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

