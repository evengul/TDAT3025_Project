import torch
import numpy as np

noise = torch.load('noise.pt')

print(noise.shape)

n = 7

Q = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

directions = torch.bucketize(torch.rand((n, n)), torch.tensor([0.33, 0.67, 1]))

other_noise =




