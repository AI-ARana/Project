from flask import Flask, render_template, request
import pennylane as qml
from pennylane import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import numpy as np

weights = np.random.rand(3, 4)  # Assuming 3 layers and 4 qubits

# Save the weights to a file
with open(r'F:\VSCode\PythonDev\Python\Sentiment\quantum_weights.pkl', 'wb') as f:
    pickle.dump(weights, f)

# Initialize PennyLane quantum device
dev = qml.device("default.qubit", wires=4)

# Load the saved model and vectorizer from the specified path
try:
    model = pickle.load(open(r'F:\VSCode\PythonDev\Python\Sentiment\sentiment_model.pkl', 'rb'))
    vectorizer = pickle.load(open(r'F:\VSCode\PythonDev\Python\Sentiment\tfidf_vectorizer.pkl', 'rb'))
    weights = pickle.load(open(r'F:\VSCode\PythonDev\Python\Sentiment\quantum_weights.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure that all required files are in the correct path.")
    exit()

# Quantum layer for sentiment analysis
@qml.qnode(dev)
def quantum_layer(inputs, weights):
    # Apply AngleEmbedding with input data
    qml.templates.AngleEmbedding(inputs, wires=range(4))
    # Apply BasicEntanglerLayers with weights as parameters
    qml.templates.BasicEntanglerLayers(weights=weights, wires=range(4))
    # Return expectation values for each qubit
    return [qml.expval(qml.PauliZ(wire)) for wire in range(4)]

# Flask app initialization
app = Flask(__name__)

# Helper function for sentiment labeling
def get_sentiment_label(prediction_score):
    # Adjusted thresholds for better results
    if prediction_score > 0.6:  # Higher positive threshold
        sentiment = 'Positive 😊'
    elif prediction_score < -0.6:  # Lower negative threshold
        sentiment = 'Negative 😔'
    else:
        sentiment = 'Neutral 😐'
    return sentiment, prediction_score

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
    try:
        if request.method == 'POST':
                user_input = request.form['text']
                cleaned_input = ' '.join(re.findall(r'\b\w+\b', user_input.lower()))

                # Vectorize user input text using the pre-trained vectorizer
                user_vector = vectorizer.transform([cleaned_input]).toarray()

                # Check if user vector has at least 4 features for quantum processing
                if user_vector.shape[1] < 4:
                    raise ValueError("The vectorized input does not have enough features for the quantum model.")

                # Use only the first 4 features for input to the quantum model
                user_vector = user_vector[:, :4]

                # Quantum prediction
                quantum_output = quantum_layer(user_vector[0], weights)  # Use the first row for prediction
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
                <h3>Sentiment: {sentiment[0]}</h3>
                <p>Score: {sentiment[1]:.2f}</p>
                <br>
                <form action="/" method="GET">
                    <button type="submit">Try Again</button>
                </form>
                </body>
                </html>
                '''

    except Exception as e:
        print(f"Error during prediction: {e}")
        return '''
        <h3>An error occurred during prediction. Please try again later.</h3>
        <a href="/">Go Back</a>
        '''

if __name__ == '__main__':
    app.run(debug=False)
