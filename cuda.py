import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")

# import torch
# print(torch.cuda.is_available())  # Should print True if CUDA is available
# print(torch.cuda.get_device_name(0))  # Should print the name of your GPU

