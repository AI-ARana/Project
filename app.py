from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = None

@app.route('/upload', methods=['POST'])
def upload_data():
    global model
    data = request.get_json()
    teams_data = data.get('teamsData', [])

    if not teams_data:
        return jsonify({'error': 'No data provided'}), 400

    # Convert the data to a DataFrame
    df = pd.DataFrame(teams_data)

    # Ensure the correct columns exist
    required_columns = {'round', 'criterion1', 'criterion2', 'criterion3'}
    if not required_columns.issubset(df.columns):
        return jsonify({'error': 'Missing required columns'}), 400

    # Prepare features and target
    X = df[['round', 'criterion1', 'criterion2']]
    y = df['criterion3']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    global model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return jsonify({'status': 'Model trained successfully'})

def predict_score(round, criterion1, criterion2):
    if model is None:
        return None
    features = [[round, criterion1, criterion2]]
    prediction = model.predict(features)
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not trained yet'}), 400
    
    data = request.get_json()
    round = data.get('round')
    criterion1 = data.get('criterion1')
    criterion2 = data.get('criterion2')
    
    if round is None or criterion1 is None or criterion2 is None:
        return jsonify({'error': 'Missing required parameters'}), 400

    predicted_score = predict_score(round, criterion1, criterion2)
    return jsonify({'predicted_score': predicted_score})

if __name__ == '__main__':
    app.run(debug=False)
