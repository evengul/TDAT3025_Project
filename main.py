import glob

from PIL import Image, ImageDraw
import json
import numpy as np
import os
import random
import torch
from torchvision import transforms

import time

from PersonalImage import PersonalImage
from TrainedModel import TrainedModel
from Noise import Noise

'''
Retrieve the image to be classified from a file path
@:param path[str]: The path where the image is located
'''


def get_x(path):
    x = Image.open(path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    return preprocess(x).type(torch.FloatTensor)


# Get an image from a tensor
def get_image(x):
    transform = transforms.ToPILImage(mode='RGB')
    return transform(x)


'''
The class containing methods to create and run the adversarial model.
@:param x_paths_train[str]: All training image paths
@:param x_paths_test[str]: All test image paths
@:param orthogonal_directions[float[][]]: Contains all directions the model can train in
@:param learning_rate[float]: How much to change a pixel for each training. Between 0 and 1 (1 is equal to 255 in pixel value)
@:param kernel_size[int]: How many squares along the side of the noise?
@:param noise_path[str]: Where an existing noise.pt file can be found
'''


class AML:
    def __init__(self, x_paths_train, x_paths_test, orthogonal_directions, learning_rate, kernel_size, noise_path=None):
        self.internal_model = TrainedModel()
        self.x_paths_train = x_paths_train
        self.x_paths_test = x_paths_test
        self.Q = orthogonal_directions
        self.learning_rate = learning_rate
        self.noise = Noise(kernel_size, 3, 224, noise_path)
        if torch.cuda.is_available():
            self.noise.switch_device('cuda')

    '''
        f: Predictor
        :param x[Tensor]: image
        :param c_0[int]: Original classification
        :param p_0[float]: Original classification percentage
    '''

    def f(self, x, c_0, p_0):
        c_1, p_1 = c_0, p_0
        allIs = list(range(len(self.noise.noise)))
        flipped = False     # Have we misclassified this image?
        # For all tiles in the noise
        for i in allIs:
            for j in allIs:
                q = random.choice(self.Q)  # Choose a vector to create the difference along (R, G or B)
                # For every direction on this vector, with the size of the learning rate
                for a in [self.learning_rate, -self.learning_rate]:
                    # Make a copy of the noise
                    possibly_changing_noise = self.noise.noise.detach().clone()
                    # Change the noise copy with the randomly chosen vector in direction a
                    possibly_changing_noise[i][j][0] += a * q[0]
                    possibly_changing_noise[i][j][1] += a * q[1]
                    possibly_changing_noise[i][j][2] += a * q[2]
                    # Get the new class and probability of the direction along the chosen vector
                    x += self.noise.to_image_tensor('cuda', noise=possibly_changing_noise)
                    c_1, p_1 = self.internal_model.run(x)
                    # If we haven't flipped yet, and do so now, we can flip
                    if not flipped and c_1 != c_0:
                        flipped = True
                    # If we have flipped and still are flipped, and the percentage is larger than it was before,
                    # perform the change.
                    if flipped and c_1 != c_0 and p_1.item() > p_0.item():
                        self.noise.noise = possibly_changing_noise
                        break
                    # Otherwise, try the other way on this vector.
                    # If that doesn't work, we choose a new random vector
                # Smoothen the noise
                self.noise.noise = torch.tanh_(self.noise.noise)
        return c_1, p_1

    # Accuracy
    def accuracy(self, eps):
        files = self.x_paths_test
        total = len(files) * 1.0
        correct = 0.0
        i = 1
        for path in files:
            if i % 100 == 0:
                print("%.2f percent done. Accuracy: %.2f percent" % ((i / total) * 100, ((correct / i) * 100)))
            i += 1
            img = get_x(path)
            c0, p0, c1, p1, x = self.apply_changes(img, eps)
            if c0 != c1 and p1 >= 0.85 * p0 or p1 <= 0.1 * p0:  # or p1 <= x * p0
                correct += 1.0
        return correct / total

    # Train AML model
    def train(self, n_runs=50, n_img=500):
        i = 0
        # files = random.sample(self.x_paths_train.tolist(), n_img) # Select random images to train on
        files = self.x_paths_train[0: n_img]    # Select images from the same class to train on

        start_time = time.time()

        previous_time = start_time

        per_round = 0
        estimated_time = 0

        # For every image:
        for file_path in files:
            # Make a tensor of the image
            x = get_x(file_path)
            if torch.cuda.is_available():
                x = x.to(device='cuda')
            # Get the original classification and its percentage. This is the y-value for our model, our (anti-)target
            c0, p0 = self.internal_model.run(x)
            # For every run on this image
            for j in range(n_runs):
                # Train on this image
                self.f(x, c0, p0)
            # All of this is to keep track of how long is left of training
            if i % 10 == 0 and i != 0:
                print("img #%i: %.2f percent done training..." % (i, (i / n_img) * 100))
                seconds = int(estimated_time - i * per_round)
                minutes = 0
                hours = 0
                if seconds >= 60:
                    minutes = int(seconds / 60)
                    seconds = seconds - 60 * minutes
                if minutes >= 60:
                    hours = int(minutes / 60)
                    minutes = minutes - 60 * hours
                print("Time left: %ih,%im,%is" % (hours, minutes, seconds))
            i += 1
            if i == 1:
                per_round = time.time() - previous_time
                estimated_time = per_round * n_img
                print("Estimated time: " + str(int(estimated_time)) + "seconds")
                previous_time = time.time()

    # Save an image after training with a specific epsilon, classification and percentage
    def save_changed_img(self, img, name, eps, c, p):
        img = get_image(img)
        d = ImageDraw.Draw(img)
        d.text((10, 10), "eps=%.4f, class=%s, p=%.2f percent" % (eps, c, p * 100), fill=(255, 0, 0))
        img.save("images/after_noise/week3/result_" + name)

    # Run model on image
    # compare actual output from google model with ours
    def apply_changes(self, x, eps):
        c_0, p_0 = self.internal_model.run(x)
        change = self.noise.to_image_tensor('cpu') * eps
        x = (x + change).clamp(0, 1)
        c_1, p_1 = self.internal_model.run(x)
        return c_0, p_0, c_1, p_1, x


# Load labels for the classes
def load_labels():
    classes_file = open('labels.json')
    loaded_labels = json.load(classes_file)
    classes_file.close()
    return loaded_labels


# Get all the paths for all train images
def retrieve_train_paths():
    path = 'images/imagenet/train'
    dirs = os.listdir(path)
    files = []
    for _dir in dirs:
        dir_path = path + "/" + _dir + "/images"
        files_paths = os.listdir(dir_path)
        for file_path in files_paths:
            files.append(dir_path + "/" + file_path)
    return np.array(files).flatten()


# Get all paths for all test images
def retrieve_test_paths():
    path = 'images/imagenet/test/images'
    return [path + "/" + file_path for file_path in os.listdir(path)]


# Run the model on one image (without training)
def run_personal(path, model, eps):
    img_obj = PersonalImage(path)
    x = img_obj.x.cpu()
    c_true, p_true, c_adv, p_adv, x = model.apply_changes(x, eps)
    labels = load_labels()
    print("eps=%.4f" % eps)
    print(
        "Original: class '" + labels[str(c_true)] + "' with " + "{:.3f}".format(p_true.item() * 100) + "% probability")
    print(
        "After AML: class '" + labels[str(c_adv)] + "' with " + "{:.3f}".format(p_adv.item() * 100) + "% probability")
    model.save_changed_img(x.cpu(), path.split("/")[-1].split(".")[0] + "_eps_%.4f.jpg" % eps,
                           eps, labels[str(c_adv)], p_adv)
    print("-----------------")


# Make a gif of all images in the folder containing images produced by run_personal
def make_gif(name):
    out_path = "images/gifs/%s.gif" % name
    files = os.listdir("images/after_noise/week3")
    files.reverse()
    img, *images = [Image.open("images/after_noise/week3/" + f) for f in files]
    img.save(fp=out_path, format="GIF", append_images=images, save_all=True, duration=500, loop=0)


# The orthogonal vector defining our directions
Q = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
if torch.cuda.is_available():
    Q = Q.to(device='cuda')

# The model we want to train on, loaded with preexisting noise
model = AML(retrieve_train_paths(), retrieve_test_paths(), Q, 0.1, 7, "noise.pt")

# Train the model
model.train(n_runs=1, n_img=10)
model.noise.save_noise('RGB')
model.noise.get_stats()

# 0.4 is chosen from testing the code below
print(model.accuracy(0.4))

# Uncomment to see the change in percentages for various eps-values
# for eps in range(1, 10):
#     eps = eps / 10000
#     run_personal("images/personal/dog.jpg", model, eps)
#     run_personal(hourglassPath, model, eps)
#
# for eps in range(1, 10):
#     eps = eps / 1000
#     run_personal("images/personal/dog.jpg", model, eps)
#     run_personal(hourglassPath, model, eps)
#
# for eps in range(1, 10):
#     eps = eps / 100
#     run_personal("images/personal/dog.jpg", model, eps)
#     run_personal(hourglassPath, model, eps)
#
# for eps in range(1, 10):
#     eps = eps / 10
#     run_personal("images/personal/dog.jpg", model, eps)
#     run_personal(hourglassPath, model, eps)

