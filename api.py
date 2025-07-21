from flask import Flask, jsonify, send_from_directory
from flask import request
from flask_cors import CORS
from tag_prediction import suggest_tags

# Enable CORS for the Flask app
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api', methods=['GET'])
def api_route():
    return jsonify({"message": "Hello World!"})


@app.route('/api/predict-tags', methods=['POST'])
def predict_tags_route():
    # Get content from the request body
    data = request.get_json()
    title = data.get('title', '')
    body = data.get('body', '')
    
    tags = suggest_tags(title, body)
    
    # Use the variables in the response for demonstration
    return jsonify({
        "tags": tags,   
    })

import os

PORT = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=PORT)
