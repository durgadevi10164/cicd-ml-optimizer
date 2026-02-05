#!/usr/bin/env python3
"""
CI/CD ML Optimizer Dashboard
Flask web application for build optimization with AI Chat
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
import openai
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.predict import predict_build_time
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# Load environment variables and setup OpenAI
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

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
            'by_branch': df.groupby('branch_type').agg({
                'build_time_minutes': 'mean',
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
    """Chat with AI about build optimization"""
    try:
        data = request.json
        user_message = data.get('message', '')
        build_context = data.get('context', {})
        conversation_history = data.get('history', [])
        
        # Create context about the user's build
        context_prompt = f"""
You are an expert CI/CD build optimization assistant. Help users understand their build performance.

CURRENT BUILD CONFIGURATION:
- Repository Size: {build_context.get('repo_size_mb', 'N/A')} MB
- Number of Files: {build_context.get('num_files', 'N/A')}
- Test Count: {build_context.get('test_count', 'N/A')}
- Dependencies: {build_context.get('dependencies', 'N/A')}
- Branch Type: {build_context.get('branch_type', 'N/A')}
- Cache Hit Rate: {build_context.get('cache_hit_rate', 'N/A')}
- Current Resources: {build_context.get('cpu_cores', 'N/A')} cores / {build_context.get('memory_gb', 'N/A')} GB

RECOMMENDATION GIVEN:
- Recommended Resources: {build_context.get('recommended_cpu', 'N/A')} cores / {build_context.get('recommended_memory', 'N/A')} GB
- Predicted Build Time: {build_context.get('predicted_time', 'N/A')} minutes
- Estimated Cost: ${build_context.get('estimated_cost', 'N/A')}
- Savings: {build_context.get('savings_percent', 'N/A')}%

Your job:
1. Answer questions about WHY the build is slow/expensive
2. Explain the optimization recommendations
3. Provide actionable tips to improve performance
4. Be specific and reference the actual numbers above
5. Keep responses concise (3-4 sentences max unless asked for details)

Be friendly, technical, and helpful!
"""
        
        # Build messages for OpenAI
        messages = [
            {"role": "system", "content": context_prompt}
        ]
        
        # Add conversation history
        for msg in conversation_history:
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        return jsonify({
            'success': True,
            'response': ai_response
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
    print(f"\n🤖 AI Chat: Enabled (OpenAI GPT-3.5)")
    print("\n⚡ Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
