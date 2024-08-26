import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Generate a simple sequence of numbers
def generate_sequence(length):
    return np.array([i for i in range(length)])

sequence = generate_sequence(100)  # Generate a sequence from 0 to 99

print("Sequence:", sequence)

def create_dataset(sequence, look_back=1):
    X, y = [], []
    for i in range(len(sequence) - look_back):
        X.append(sequence[i:(i + look_back)])
        y.append(sequence[i + look_back])
    return np.array(X), np.array(y)

look_back = 3
X, y = create_dataset(sequence, look_back)

print("Input data shape:", X.shape)  # Should be (samples, time steps)
print("Output data shape:", y.shape)


# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print("Reshaped Input data shape:", X.shape)  # Should be (samples, time steps, features)


model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))  # 50 LSTM units
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mse')

print(model.summary())

# Train the model
model.fit(X, y, epochs=100, batch_size=1, verbose=2)

# Predict the next value in the sequence
last_sequence = sequence[-look_back:]  # Take the last `look_back` values
last_sequence = np.reshape(last_sequence, (1, look_back, 1))  # Reshape for prediction
predicted_value = model.predict(last_sequence)
print("Predicted next value in the sequence:", predicted_value[0][0])

# Evaluate the model's prediction
actual_next_value = sequence[-1] + 1  # Since it's a simple sequence, we know the next value
print("Actual next value in the sequence:", actual_next_value)
print("Error in prediction:", abs(predicted_value[0][0] - actual_next_value))
