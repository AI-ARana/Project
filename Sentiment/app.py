from flask import Flask, render_template, request
import pickle
import re
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load the saved model and vectorizer from 'model/' folder
model = pickle.load(open(r'F:\VSCode\PythonDev\Python\Sentiment\sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open(r'F:\VSCode\PythonDev\Python\Sentiment\tfidf_vectorizer.pkl', 'rb'))

app = Flask(__name__)

def get_sentiment_label(prediction):
    return 'Positive 😊' if prediction == 1 else 'Neutral 😐' if prediction == 0 else 'Negative 😔'

@app.route('/')
def home():
    return render_template('index.html'), 200, {'Content-Type': 'text/html'}

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("Predict route triggered")  # Debugging
        user_input = request.form['text']
        print(f"User Input: {user_input}")  # Debugging

        cleaned_input = [' '.join(re.findall(r'\b\w+\b', user_input.lower()))]
        input_vector = vectorizer.transform(cleaned_input).toarray()
        prediction = model.predict(input_vector)[0]
        sentiment = get_sentiment_label(prediction)

        print(f"Prediction: {sentiment}")  # Debugging
        return render_template('index.html', prediction_text=f'Sentiment: {sentiment}'), 200, {'Content-Type': 'text/html'}

if __name__ == '__main__':
     app.run(debug=True)
