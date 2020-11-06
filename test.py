import matplotlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

transt = transforms.ToTensor()
transp = transforms.ToPILImage()
img_t = transt(Image.open('images/personal/dog.jpg'))

#torch.Tensor.unfold(dimension, size, step)
#slices the images into 8*8 size patches
patches = img_t.data.unfold(0, 3, 3).unfold(1, 16, 16).unfold(2, 16, 16)


print(patches[0][0][0].shape)


def visualize(patches):
    """Imshow for Tensor."""
    fig = plt.figure(figsize=(4, 4))
    for i in range(4):
        for j in range(4):
            inp = transp(patches[0][i][j])
            inp = np.array(inp)

            ax = fig.add_subplot(4, 4, ((i*4)+j)+1, xticks=[], yticks=[])
            plt.imsave("%i%i_.png" % (i, j), inp)


visualize(patches)
