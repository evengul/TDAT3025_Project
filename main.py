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
from Noise import Noise, read_noise

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


def get_image(x):
    transform = transforms.ToPILImage(mode='RGB')
    return transform(x)


'''
The class containing methods to create and run the adversarial model.
@:param x_paths_train[str]: All training image paths
@:param x_paths_test[str]: All test image paths
@:param orthogonal_directions
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

        self.softmax = torch.nn.Softmax()

    # Predictor
    def f(self, x, c_0, p_0):
        c_1, p_1 = c_0, p_0
        allIs = list(range(len(self.noise.noise)))
        # Possible: Create q of l x l. Add this to noise for each l. Keep it if its good
        flipped = False
        for i in allIs:
            for j in allIs:
                q = random.choice(self.Q)  # Choose a vector to create the difference along (R, G or B)
                # For every direction on this vector, with the size of the learning rate
                for a in [self.learning_rate, -self.learning_rate]:
                    possibly_changing_noise = self.noise.noise.detach().clone()
                    # Get the new probability of the direction along the chosen vector
                    # CHANGE NOISE
                    possibly_changing_noise[i][j][0] += a * q[0]
                    possibly_changing_noise[i][j][1] += a * q[1]
                    possibly_changing_noise[i][j][2] += a * q[2]
                    x += self.noise.to_image_tensor('cuda', noise=possibly_changing_noise)
                    c_1, p_1 = self.internal_model.run(x)
                    # If we have a lower probability of getting the class we want, we go this way.
                    # Otherwise, try the other way on this vector.
                    # If that doesn't work, we choose a new random vector
                    if not flipped and c_1 != c_0:
                        flipped = True
                    if flipped and c_1 != c_0 and p_1.item() > p_0.item():
                        self.noise.noise = possibly_changing_noise
                        break
                self.noise.noise = torch.tanh_(self.noise.noise)
        return c_1, p_1

    # Accuracy
    def accuracy(self, eps):
        print("Checking accuracy...")
        files = self.x_paths_test
        total = len(files) * 1.0
        correct = 0.0
        i = 1
        for path in files:
            # if i % 100 == 0:
            #     print("%.2f percent done. Accuracy: %.2f percent" % ((i / total) * 100, ((correct / i) * 100)))
            i += 1
            img = get_x(path)
            c0, p0, c1, p1, x = self.apply_changes(img, eps)
            if c0 != c1 and p1 >= 0.85 * p0 or p1 <= 0.1 * p0:  # or p1 <= x * p0
                correct += 1.0
        return correct / total

    # Train AML model
    def train(self, n_runs=50, n_img=500):
        i = 0
        # files = random.sample(self.x_paths_train.tolist(), n_img)
        files = self.x_paths_train[0: n_img]
        probs = torch.zeros((len(files), n_runs, 2))

        start_time = time.time()

        previous_time = start_time

        per_round = 0
        estimated_time = 0

        for file_path in files:
            x = get_x(file_path)
            if torch.cuda.is_available():
                x = x.to(device='cuda')
            c0, p0 = self.internal_model.run(x)  # this is our y
            for j in range(n_runs):
                c1, p1 = self.f(x, c0, p0)
                probs[i, j] = torch.tensor([j, p1])
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
        return probs

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


def load_labels():
    classes_file = open('labels.json')
    loaded_labels = json.load(classes_file)
    classes_file.close()
    return loaded_labels


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


def retrieve_test_paths():
    path = 'images/imagenet/test/images'
    return [path + "/" + file_path for file_path in os.listdir(path)]


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


def make_gif(name):
    out_path = "images/gifs/%s.gif" % name
    files = os.listdir("images/after_noise/week3")
    files.reverse()
    img, *images = [Image.open("images/after_noise/week3/" + f) for f in files]
    img.save(fp=out_path, format="GIF", append_images=images, save_all=True, duration=500, loop=0)


Q = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
if torch.cuda.is_available():
    Q = Q.to(device='cuda')


model = AML(retrieve_train_paths(), retrieve_test_paths(), Q, 0.1, 14)

BEST_TRAIN_PROB = 0.9974
BEST_TRAIN_INDEX = 54898

EPS = 0.439

probs = model.train(n_runs=10, n_img=50)
model.noise.save_noise('RGB')

model.noise.get_stats()

hourglassPath = model.x_paths_train[BEST_TRAIN_INDEX]

for eps in range(1, 10):
    eps = eps / 10000
    run_personal("images/personal/dog.jpg", model, eps)
    run_personal(hourglassPath, model, eps)

for eps in range(1, 10):
    eps = eps / 1000
    run_personal("images/personal/dog.jpg", model, eps)
    run_personal(hourglassPath, model, eps)

for eps in range(1, 10):
    eps = eps / 100
    run_personal("images/personal/dog.jpg", model, eps)
    run_personal(hourglassPath, model, eps)

for eps in range(1, 10):
    eps = eps / 10
    run_personal("images/personal/dog.jpg", model, eps)
    run_personal(hourglassPath, model, eps)

