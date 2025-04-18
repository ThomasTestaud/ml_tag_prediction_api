from flask import Flask, jsonify
from flask import request
from flask_cors import CORS
# Enable CORS for the Flask app
app = Flask(__name__)
CORS(app)

# Import du vectorizer
import pickle
with open('CountVectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
# Import du modèle
with open('SGDClassifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Import du mlb
with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

# Fonction pour prédire les tags
import numpy as np
def suggest_tags(title, body, vectorizer, model, mlb, threshold=0.05, top_n=5):
    # Prétraitement du texte
    text = title + " " + body
    text_vectorized = vectorizer.transform([text])  # Transformer avec le même vectorizer entraîné

    # Prédiction des probabilités des classes
    y_pred_proba = model.predict_proba(text_vectorized)

    # Si la sortie est une liste (cas multi-label), extraire la probabilité de la classe positive
    if isinstance(y_pred_proba, list):
        y_pred_proba = np.array([proba[:, -1] if proba.shape[1] > 1 else proba[:, 0] for proba in y_pred_proba]).T

    # Extraire les indices des tags au-dessus du seuil
    probas = y_pred_proba[0]
    above_threshold = [(mlb.classes_[i], probas[i]) for i in range(len(probas)) if probas[i] > threshold]

    # Trier par probabilité décroissante et prendre les top_n
    above_threshold_sorted = sorted(above_threshold, key=lambda x: x[1], reverse=True)[:top_n]

    # Extraire uniquement les noms des tags
    predicted_tags = [tag for tag, _ in above_threshold_sorted]

    return predicted_tags

# Fonction pour prétraiter un corps de texte
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
def preprocess_body(body):
    # Analyser le HTML
    soup = BeautifulSoup(body, "html.parser")
    
    # Supprimer les balises <code> et <pre> (ainsi que leur contenu)
    for tag in soup.find_all(["code", "pre"]):
        tag.extract()
    
    # Extraire le texte sans balises HTML
    text = soup.get_text(separator=" ")
    
    # Supprimer les caractères spéciaux et les chiffres
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    
    # Supprimer les stopwords et appliquer la lemmatisation
    tokens = [
        lemmatizer.lemmatize(word.lower()) 
        for word in text.split() 
        # Si le mot n'est pas un stopword et est présent dans le dictionnaire
        if word.lower() not in stopwords_with_extras
    ]
    
    return " ".join(tokens)

# Ensemble de mots vides avec des mots supplémentaires
stopwords_with_extras = stopwords.union({'using', 'use', 'file', 'get'})

# Fonction pour prétraiter un titre
def preprocess_title(title):
    tokens = [
        lemmatizer.lemmatize(word.lower()) 
        #stemmer.stem(word.lower()) 
        for word in re.findall(r'[A-Za-z0-9#]+', title)  # Correction de la regex
        
        # Si le mot n'est pas un stopword et n'est pas un chiffre
        if word.lower() not in stopwords_with_extras and not word.isdigit()
    ]
    return " ".join(tokens)


@app.route('/api', methods=['GET'])
def api_route():
    return jsonify({"message": "Hello World!"})


@app.route('/api/predict-tags', methods=['POST'])
def predict_tags_route():
    # Get content from the request body
    data = request.get_json()
    title = data.get('title', '')
    body = data.get('body', '')
    
    tags = suggest_tags(title, body, vectorizer, model, mlb)
    
    # Use the variables in the response for demonstration
    return jsonify({
        "tags": tags,   
    })

if __name__ == '__main__':
    app.run(debug=True)