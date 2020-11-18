import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def read_noise(path):
    img = Image.open(path)
    return transforms.PILToTensor()(img)


class Noise:
    def __init__(self, n_rows, channels, image_size, init_noise=None):
        self.mode = 'cpu'
        self.channels = channels
        self.image_size = image_size
        self.square_size = int(image_size / n_rows)
        if init_noise is not None:
            self.noise = torch.load(init_noise)
        else:
            if image_size % n_rows == 0:
                self.noise = (torch.rand((n_rows, n_rows, channels, self.square_size, self.square_size))) * 2 - 1
            else:
                raise ValueError("image_size must be divisible by n_rows!")

    def switch_device(self, device):
        if device == 'cuda':
            self.noise = self.noise.to(device='cuda')
            self.mode = 'cuda'
        if device == 'cpu':
            self.noise = self.noise.cpu()
            self.mode = 'cpu'

    def to_image_tensor(self, device, noise=None):
        if noise is not None:
            noise_reshaped = np.transpose(noise.cpu(), (2, 0, 3, 1, 4)).reshape((3, 224, -1))
        else:
            noise_reshaped = np.transpose(self.noise.cpu(), (2, 0, 3, 1, 4)).reshape((3, 224, -1))
        if torch.cuda.is_available() and device == 'cuda':
            return noise_reshaped.to(device='cuda')
        elif device == 'cpu':
            return noise_reshaped.cpu()

    def save_noise(self, mode, name='noise'):
        img_tensor = self.to_image_tensor('cpu')
        torch.save(self.noise.cpu(), '%s.pt' % name)
        transforms.ToPILImage(mode=mode)(img_tensor).save("%s.png" % name)

    def get_stats(self):
        tiles_per_side = self.noise.shape[0]
        tile_colors = torch.zeros(tiles_per_side, tiles_per_side, 3)
        for i in range(tiles_per_side):
            for j in range(tiles_per_side):
                tile_colors[i, j, 0] = torch.mean(self.noise[i, j, 0]) * 255
                tile_colors[i, j, 1] = torch.mean(self.noise[i, j, 1]) * 255
                tile_colors[i, j, 2] = torch.mean(self.noise[i, j, 2]) * 255
        reds = tile_colors[:, :, 0]
        greens = tile_colors[:, :, 1]
        blues = tile_colors[:, :, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reds, greens, blues)
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")
        plt.show()


if __name__ == "main":
    noise = Noise(7, 3, 224, "noise.pt")

    noise.get_stats()





