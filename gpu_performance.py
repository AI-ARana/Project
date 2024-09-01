import time
import numpy as np
import torch

# Timing CPU computation
start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sequence = [0.1, 0.2, 0.3]
input_size = 1
hidden_size = 1
output_size = 1

W_xh = np.random.randn(hidden_size, input_size)
W_hh = np.random.randn(hidden_size, hidden_size)
W_hy = np.random.randn(output_size, hidden_size)
b_h = np.zeros((hidden_size, 1))
b_y = np.zeros((output_size, 1))

h_t = np.zeros((hidden_size, 1))

def rnn_step(x_t, h_t):
    h_t = np.tanh(np.dot(W_xh, x_t) + np.dot(W_hh, h_t) + b_h)
    y_t = np.dot(W_hy, h_t) + b_y
    return y_t, h_t

for x_t in sequence:
    x_t = np.array([[x_t]])
    y_t, h_t = rnn_step(x_t, h_t)
    print(f"Input: {x_t.flatten()}, Predicted Output: {y_t.flatten()}")

cpu_time = time.time() - start_time
print(f"CPU Time: {cpu_time} seconds")

# Timing GPU computation
if torch.cuda.is_available():
    start_time = time.time()

    # Convert numpy arrays to torch tensors
    W_xh = torch.tensor(W_xh, device=device, dtype=torch.float32)
    W_hh = torch.tensor(W_hh, device=device, dtype=torch.float32)
    W_hy = torch.tensor(W_hy, device=device, dtype=torch.float32)
    b_h = torch.tensor(b_h, device=device, dtype=torch.float32)
    b_y = torch.tensor(b_y, device=device, dtype=torch.float32)
    h_t = torch.tensor(h_t, device=device, dtype=torch.float32)

    def rnn_step(x_t, h_t):
        h_t = torch.tanh(torch.mm(W_xh, x_t) + torch.mm(W_hh, h_t) + b_h)
        y_t = torch.mm(W_hy, h_t) + b_y
        return y_t, h_t

    for x_t in sequence:
        x_t = torch.tensor([[x_t]], device=device, dtype=torch.float32)
        y_t, h_t = rnn_step(x_t, h_t)
        print(f"Input: {x_t.cpu().numpy().flatten()}, Predicted Output: {y_t.cpu().numpy().flatten()}")

    gpu_time = time.time() - start_time
    print(f"GPU Time: {gpu_time} seconds")
else:
    print("CUDA is not available. GPU timing skipped.")
