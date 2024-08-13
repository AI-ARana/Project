import numpy as np

# Initialize weights and biases
W_xh = 0.5  # Weight for input to hidden
W_hh = 0.8  # Weight for hidden to hidden
W_hy = 1.0  # Weight for hidden to output

b_h = 0.0   # Bias for hidden
b_y = 0.0   # Bias for output

# Initialize the hidden state
h_t = 0.0

# Input sequence
sequence = [0.1, 0.2, 0.3]

# Function to calculate the hidden state and output
def rnn_step(x_t, h_t):
    h_t = np.tanh(W_xh * x_t + W_hh * h_t + b_h)
    y_t = W_hy * h_t + b_y
    return y_t, h_t

# Iterate over the sequence
for i, x_t in enumerate(sequence):
    y_t, h_t = rnn_step(x_t, h_t)
    print(f"Time Step {i+1}: Input: {x_t}, Hidden State: {h_t:.5f}, Output: {y_t:.5f}")
