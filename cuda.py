# import torch
#
# print("CUDA available:", torch.cuda.is_available())
# print("Number of GPUs:", torch.cuda.device_count())
# print("Current device:", torch.cuda.current_device())
# print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")

# import torch
# print(torch.__version__)
# print(torch.version.cuda)


# import torch
# if torch.cuda.is_available():
#     print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
# else:
#     print("CUDA is not available. Using CPU.")

import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
