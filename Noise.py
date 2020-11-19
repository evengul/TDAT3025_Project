import torch
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt

'''
    Noise: The class defining the noise layer that can be added to the images in the model.
'''


class Noise:
    def __init__(self, n_rows, channels, image_size, init_noise=None):
        self.mode = 'cpu'                               # Are we running on GPU or CPU?
        self.channels = channels                        # How many channels are in the images?
        self.image_size = image_size                    # How many pixels are along the sides of the square image?
        self.square_size = int(image_size / n_rows)     # How big are the tiles of the noise?
        if init_noise is not None:                      # If we have some initial noise.pt file
            self.noise = torch.load(init_noise)         # Load this
        else:                                           # If we don't
            if image_size % n_rows == 0:
                # Initialise the noise randomly
                self.noise = (torch.rand((n_rows, n_rows, channels, self.square_size, self.square_size))) * 2 - 1
            else:
                raise ValueError("image_size must be divisible by n_rows!")

    # Set the used device to another
    def switch_device(self, device):
        if device == 'cuda':
            self.noise = self.noise.to(device='cuda')
            self.mode = 'cuda'
        if device == 'cpu':
            self.noise = self.noise.cpu()
            self.mode = 'cpu'

    # Convert the noise to an image shape (3,224,224)
    def to_image_tensor(self, device, noise=None):
        if noise is not None:
            noise_reshaped = np.transpose(noise.cpu(), (2, 0, 3, 1, 4)).reshape((3, 224, -1))
        else:
            noise_reshaped = np.transpose(self.noise.cpu(), (2, 0, 3, 1, 4)).reshape((3, 224, -1))
        if torch.cuda.is_available() and device == 'cuda':
            return noise_reshaped.to(device='cuda')
        elif device == 'cpu':
            return noise_reshaped.cpu()

    # Save the noise.pt file
    def save_noise(self, mode, name='noise'):
        img_tensor = self.to_image_tensor('cpu')
        torch.save(self.noise.cpu(), '%s.pt' % name)
        transforms.ToPILImage(mode=mode)(img_tensor).save("%s.png" % name)

    # Get the statistics of the image (3d-plotted points in RGB-space)
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

