import torch
from torchvision import transforms
import numpy as np

noise = torch.zeros((7, 7, 3, 32, 32))

for i in range(len(noise)):
    for j in range(len(noise[i])):
        if (i % 2 == 0 and j % 2 != 0) or (i % 2 != 0 and j % 2 == 0):
            noise[i][j] = torch.ones((3, 32, 32))

t = transforms.ToPILImage(mode='RGB')

noise_reshaped = torch.zeros((3, 224, 224))

for i in range(len(noise)):
    for j in range(len(noise[i])):
        noise_reshaped[:, i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = noise[i, j]

noise_reshaped_2 = np.transpose(noise, (2, 0, 3, 1, 4)).reshape((3, 224, -1))


t(noise_reshaped).save("something.png")
t(noise_reshaped_2).save("something2.png")


