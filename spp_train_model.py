import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# Sample training data (grades and attendance)
X = np.array([
    [80, 85, 90, 95],
    [60, 65, 70, 75],
    [50, 55, 60, 65],
    [70, 75, 80, 85]
])

# Target (final grades)
y = np.array([92, 76, 66, 84])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'student_model.pkl')
