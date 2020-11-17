import torch
from torchvision import transforms
import numpy as np


class Noise:
    def __init__(self, n_rows, channels, image_size, init_noise=None):
        self.mode = 'cpu'
        self.channels = channels
        self.image_size = image_size
        self.square_size = int(image_size / n_rows)
        if init_noise is not None:
            self.noise = init_noise
        else:
            if image_size % n_rows == 0:
                self.noise = (torch.zeros((n_rows, n_rows, channels, self.square_size, self.square_size)))
            else:
                raise ValueError("image_size must be divisible by n_rows!")

    def switch_device(self, device):
        if device == 'cuda':
            self.noise = self.noise.to(device='cuda')
            self.mode = 'cuda'
        if device == 'cpu':
            self.noise = self.noise.cpu()
            self.mode = 'cpu'

    def to_image_tensor(self, device):
        noise_reshaped = np.transpose(self.noise.cpu(), (2, 0, 3, 1, 4)).reshape((3, 224, -1))
        if torch.cuda.is_available() and device == 'cuda':
            return noise_reshaped.to(device='cuda')
        elif device == 'cpu':
            return noise_reshaped.cpu()

    def save_noise_image(self, mode, name='noise'):
        transforms.ToPILImage(mode=mode)(self.to_image_tensor('cpu')).save("%s.png" % name)


