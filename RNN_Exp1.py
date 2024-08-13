import numpy as np

# Define the sequence and parameters
sequence = [0.1, 0.2, 0.3]  # Example sequence
input_size = 1
hidden_size = 1
output_size = 1

# Initialize weights and biases
W_xh = np.random.randn(hidden_size, input_size)  # Input to hidden
W_hh = np.random.randn(hidden_size, hidden_size)  # Hidden to hidden
W_hy = np.random.randn(output_size, hidden_size)  # Hidden to output

b_h = np.zeros((hidden_size, 1))  # Bias for hidden layer
b_y = np.zeros((output_size, 1))  # Bias for output layer

# Initialize the hidden state
h_t = np.zeros((hidden_size, 1))

# Function to calculate the next step
def rnn_step(x_t, h_t):
    h_t = np.tanh(np.dot(W_xh, x_t) + np.dot(W_hh, h_t) + b_h)
    y_t = np.dot(W_hy, h_t) + b_y
    return y_t, h_t

# Iterate over the sequence
for x_t in sequence:
    x_t = np.array([[x_t]])  # Convert input to 2D array
    y_t, h_t = rnn_step(x_t, h_t)
    print(f"Input: {x_t.flatten()}, Predicted Output: {y_t.flatten()}")
