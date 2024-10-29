from flask import Flask, render_template, request
import pennylane as qml
from pennylane import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Initialize PennyLane quantum device
dev = qml.device("default.qubit", wires=4)

# Quantum layer for sentiment analysis
@qml.qnode(dev)
def quantum_layer(inputs):
    qml.templates.AngleEmbedding(inputs, wires=range(4))
    qml.templates.BasicEntanglerLayers(weights=np.random.rand(3, 4), wires=range(4))
    return [qml.expval(qml.PauliZ(wire)) for wire in range(4)]

# Initialize TF-IDF Vectorizer with a higher maximum number of features
vectorizer = TfidfVectorizer(max_features=20)

# Flask app initialization
app = Flask(__name__)

# Helper function for sentiment labeling
def get_sentiment_label(prediction_score):
    if prediction_score > 0.3:  # Adjusted thresholds
        return 'Positive 😊'
    elif prediction_score < -0.3:
        return 'Negative 😔'
    else:
        return 'Neutral 😐'

# Route for home page
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis</title>
    </head>
    <body>
    <h2>Sentiment Analysis</h2>
    <form action="/predict" method="POST">
        <textarea name="text" rows="5" cols="50" placeholder="Enter your text..."></textarea><br><br>
        <button type="submit">Analyze Sentiment</button>
    </form>
    </body>
    </html>
    '''

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['text']
        cleaned_input = ' '.join(re.findall(r'\b\w+\b', user_input.lower()))

        # Vectorize user input text
        user_vector = vectorizer.fit_transform([cleaned_input]).toarray()

        # Ensure we only take the first 4 features for the quantum model
        if user_vector.shape[1] > 4:
            user_vector = user_vector[:, :4]

        # Quantum prediction
        quantum_output = quantum_layer(user_vector[0])  # Use the first row for prediction
        prediction_score = np.mean(quantum_output)

        # Debugging output to see quantum output and prediction score
        print(f"Quantum Output: {quantum_output}, Prediction Score: {prediction_score}")

        sentiment = get_sentiment_label(prediction_score)

        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Sentiment Analysis</title>
        </head>
        <body>
        <h2>Sentiment Analysis</h2>
        <form action="/predict" method="POST">
            <textarea name="text" rows="5" cols="50" placeholder="Enter your text..."></textarea><br><br>
            <button type="submit">Analyze Sentiment</button>
        </form>
        <h3>Sentiment: {sentiment}</h3>
        </body>
        </html>
        '''

if __name__ == '__main__':
    app.run(debug=True)
