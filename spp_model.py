import sys
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

if len(sys.argv) < 3:
    print("Error: Please provide previous grades and attendance.")
    sys.exit(1)

# Load pre-trained model
model_path= 'F:\VSCode\student_model.pkl'
model = joblib.load(model_path)

# Parse input data
grades = json.loads(sys.argv[1])
attendance = float(sys.argv[2])

# Create feature array for the model (grades + attendance)
features = np.array(grades + [attendance]).reshape(1, -1)

# Predict the future grade
predicted_grade = model.predict(features)[0]

# Output the result
print(predicted_grade)
