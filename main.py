import torch
device = 'cpu'
if torch.cuda.is_available():
    device = 'gpu:0'
print("Using device: " + str(device))
