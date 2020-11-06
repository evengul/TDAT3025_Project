from PIL import Image
import json
import numpy as np
import os
import random
import torch
from torchvision import transforms

from PersonalImage import PersonalImage
from TrainedModel import TrainedModel

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
    def __init__(self, x_paths_train, x_paths_test, orthogonal_directions, learning_rate, eps, kernel_size):
        self.internal_model = TrainedModel()
        self.x_paths_train = x_paths_train
        self.x_paths_test = x_paths_test
        self.Q = orthogonal_directions
        self.learning_rate = learning_rate
        self.eps = eps
        self.noise = torch.tensor(np.ones((kernel_size, kernel_size, 3, int(224 / kernel_size), int(224 / kernel_size)))).type(torch.FloatTensor)

    # Predictor
    def f(self, x, c_0, p_0):
        c_1, p_1 = c_0, p_0
        for i in range(len(self.noise)):
            for j in range(len(self.noise[i])):
                q = random.choice(self.Q)  # Choose a vector to create the difference along (R, G or B)
                # For every direction on this vector, with the size of the learning rate
                for a in [self.learning_rate, -self.learning_rate]:
                    # Get the new probability of the direction along the chosen vector
                    # CHANGE NOISE
                    self.noise[i][j][0] += a * q[0]
                    self.noise[i][j][1] += a * q[1]
                    self.noise[i][j][2] += a * q[2]
                    x += self.noise.reshape(3, 224, 224)
                    c_1, p_1 = self.internal_model.run(x)
                    # If we have a lower probability of getting the class we want, we go this way.
                    # Otherwise, try the other way on this vector. If that doesn't work, we choose a new random vector
                    if p_1.item() < p_0.item() or c_1 != c_0:
                        break
                    else:
                        self.noise[i][j][0] -= a * q[0]
                        self.noise[i][j][1] -= a * q[1]
                        self.noise[i][j][2] -= a * q[2]
        return c_1, p_1

    # Accuracy - TODO: Make it take smaller changes in percentage into account?
    def accuracy(self):
        total = len(self.x_paths_test) * 1.0
        correct = 0.0
        i = 1
        for path in self.x_paths_test:
            if i % 100 == 0:
                print("%.2f percent done. Accuracy: %.2f percent" % ((i / total) * 100, ((correct / i) * 100)))
            i += 1
            img = get_x(path)
            c0, p0, c1, p1 = self.apply_changes(img)
            if c0 != c1:
                correct += 1.0
        return correct / total

    # Loss
    def loss(self):
        print()

    # Train AML model
    def train(self, n_runs=50, n_img=500):
        i = 0
        for file_path in self.x_paths_train[0:n_img]:
            if i % 10 == 0:
                print("img #%i: %.2f percent done training..." % (i, (i / n_img) * 100))
            i += 1
            x = get_x(file_path)
            c0, p0 = self.internal_model.run(x)  # this is our y
            for n in range(n_runs):
                c1, _ = self.f(x, c0, p0)
                if c1 != c0:
                    break

    def save_changed_img(self, img, name):
        x = img + self.noise.reshape(3, 224, 224) * self.eps
        get_image(x).save("result_" + name)

    # Test model on another dataset
    def test(self):
        print()

    # Run model on image
    # compare actual output from google model with ours
    def apply_changes(self, x):
        c_0, p_0 = self.internal_model.run(x)
        x = x + self.noise.reshape(3, 224, 224) * self.eps
        c_1, p_1 = self.internal_model.run(x)
        return c_0, p_0, c_1, p_1


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


Q = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
model = AML(retrieve_train_paths(), retrieve_test_paths(), Q, 0.01, 1, 7)


def run_personal(path):
    img_obj = PersonalImage(path)
    c_true, p_true, c_adv, p_adv = model.apply_changes(img_obj.x)
    labels = load_labels()
    print(
        "Original: class '" + labels[str(c_true)] + "' with " + "{:.3f}".format(p_true.item() * 100) + "% probability")
    print(
        "After AML: class '" + labels[str(c_adv)] + "' with " + "{:.3f}".format(p_adv.item() * 100) + "% probability")
    model.save_changed_img(img_obj.x, path.split("/")[-1])


# x = get_x("images/imagenet/test/images/test_114.JPEG")
# new_img, _, c, p = model.f(x, 0)
# print()
model.train(n_runs=10, n_img=100)

transforms.ToPILImage(mode='RGB')(model.noise.reshape(3, 224, 224)).save('noise.png')

# model.accuracy()
run_personal("images/imagenet/test/images/test_12.JPEG")
run_personal("images/personal/dog.jpg")

# print(model.accuracy())
