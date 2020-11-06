from PIL import Image
from torchvision import transforms
import torch


class PersonalImage:
    def __init__(self, path):
        self.path = path
        self.img = crop_max_square(Image.open(path).convert('RGB')).resize((224, 224))
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        self.x = preprocess(self.img).type(torch.FloatTensor)


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

