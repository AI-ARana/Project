import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Generate synthetic data
def generate_sequence(length):
    return np.array([i for i in range(length)])

sequence = generate_sequence(100)

# Prepare data for LSTM
def create_dataset(sequence, look_back=1):
    X, y = [], []
    for i in range(len(sequence) - look_back):
        X.append(sequence[i:(i + look_back)])
        y.append(sequence[i + look_back])
    return np.array(X), np.array(y)

look_back = 3
X, y = create_dataset(sequence, look_back)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=100, batch_size=1, verbose=2)

# Predict the next value in the sequence
last_sequence = sequence[-look_back:]  # Take the last `look_back` values
last_sequence = np.reshape(last_sequence, (1, look_back, 1))
predicted_value = model.predict(last_sequence)
print("Predicted next value in the sequence:", predicted_value[0][0])
