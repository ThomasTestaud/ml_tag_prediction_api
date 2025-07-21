from flask import Flask, jsonify, send_from_directory
from flask import request
from flask_cors import CORS
# Enable CORS for the Flask app
app = Flask(__name__)
CORS(app)



# importation des bibliothèques nécessaires pour la suggestion de tags
import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib
import nltk

vectorizer = joblib.load('TfidfVectorizer.pkl')
model = joblib.load('LogisticRegression_model.pkl')
mlb = joblib.load('MultiLabelBinarizer.pkl')

# Téléchargement des stopwords et du lemmatizer si nécessaire
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = nltk.WordNetLemmatizer()
# import des dictionnaires
bodyDictionnary = joblib.load('bodyDictionnary.pkl')
titleDictionnary = joblib.load('titleDictionnary.pkl')

# Ensemble de mots vides avec des mots supplémentaires
stopwords_with_extras = set(stopwords.words('english')).union({'using', 'use', 'file', 'get', '#39', 't', 's', 'j', 'quot', 'error', 'value', 'way', 'im', 'like', 'ive'})

# Fonction pour prétraiter un titre
def preprocess_title(title):
    tokens = [
        lemmatizer.lemmatize(word.lower()) 
        #stemmer.stem(word.lower()) 
        for word in re.findall(r'[A-Za-z0-9#]+', title)  # Correction de la regex
        
        # Si le mot n'est pas un stopword et n'est pas un chiffre
        if word.lower() not in stopwords_with_extras and not word.isdigit()
        
        # Ne garder que si présent dans le dictionnaire
        and word.lower() in titleDictionnary
    ]
    return " ".join(tokens)

# Fonction pour prétraiter un corps de texte
def preprocess_body(body, lemmatize=True):
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
    if lemmatize:
        tokens = [
            lemmatizer.lemmatize(word.lower()) 
            for word in text.split() 
            if word.lower() not in stopwords_with_extras and word.lower() in bodyDictionnary
        ]
    else:
        tokens = [
            word.lower()
            for word in text.split()
            if word.lower() not in stopwords_with_extras and word.lower() in bodyDictionnary
        ]

    return " ".join(tokens)

def suggest_tags(title, body, vectorizer, model, mlb, threshold=0.05, top_n=5):
    # Prétraitement du texte
    text = preprocess_title(title)+ " " + preprocess_body(body)
    text_vectorized = vectorizer.transform([text])  # Transformer avec le même vectorizer entraîné
    
    # Importation du fichier PCA_TFIDF.pkl pour la réduction de dimension
    import joblib
    pca = joblib.load('PCA_TFIDF.pkl')
    # Réduction de dimension si nécessaire
    #if hasattr(pca, 'transform'):
    text_vectorized = pca.transform(text_vectorized)

    # Prédiction des probabilités des classes
    y_pred_proba = model.predict_proba(text_vectorized)
    print(y_pred_proba)

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
    
    tags = suggest_tags(title, body, vectorizer, model, mlb)
    
    # Use the variables in the response for demonstration
    return jsonify({
        "tags": tags,   
    })

import os

PORT = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=PORT)
