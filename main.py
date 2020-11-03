import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from matplotlib import pyplot as plt
import json
import numpy as np
import os
import random


class TrainedModel:
    def __init__(self):
        self.model = models.googlenet(pretrained=True)
        self.model.eval()
        self.filename = ""
        if torch.cuda.is_available():
            self.model.to('cuda')

    def run(self, x):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(x)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        with torch.no_grad():
            output = self.model(input_batch)
        # Returns probabilities for each class
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        classification = torch.argmax(probabilities).item()
        return classification, probabilities[classification]


'''
Retrieve the image to be classified from a file path
@:param path[str]: The path where the image is located
'''


def get_x(path):
    x = Image.open(path)
    plt.imsave("Image.open.jpg", x)
    if x.mode != 'RGB':
        return Image.fromarray(np.stack((x, x, x), axis=2))
    return x


'''
The class containing methods to create and run the adversarial model.
@:param x_paths_train[str]: All training image paths
@:param x_paths_test[str]: All test image paths
@:param orthogonal_directions
'''


class AML:
    def __init__(self, x_paths_train, x_paths_test, orthogonal_directions, learning_rate):
        self.internal_model = TrainedModel()
        self.x_paths_train = x_paths_train
        self.x_paths_test = x_paths_test
        self.Q = orthogonal_directions
        self.learning_rate = learning_rate

    # Predictor
    def f(self, x, max_runs=250):
        plt.imsave("original.jpg", Image.fromarray(np.uint8(x)))
        c_0, p_0 = self.internal_model.run(x)
        # Original guess
        c_1 = c_0
        runs = 0
        # Distance from original position
        d = [0, 0, 0]
        # While we still have the same class, or have exceeded amount of runs
        while runs < max_runs and c_1 == c_0:

            q = random.choice(self.Q)  # Choose a vector to create the difference along (R, G or B)
            # For every direction on this vector, with the size of the learning rate
            for a in [self.learning_rate, -self.learning_rate]:
                # Get the new probability of the direction along the chosen vector
                x = (np.array(x) + d + a * q).astype(int)
                c_1, p_1 = self.internal_model.run(Image.fromarray(np.uint8(x)))

                # If we have a lower probability of getting the class we want, we go this way.
                # Otherwise, try the other way on this vector. If that doesn't work, we choose a new random vector
                if p_1.item() < p_0.item() or c_1 != c_0:
                    d = d + a * q
                    p_0 = p_1
                    break

            runs += 1

        plt.imsave("changed.jpg", Image.fromarray(np.uint8(x)))

        return c_1, p_0

    # Accuracy - TODO: Make it take smaller changes in percentage into account?
    def accuracy(self):
        total = len(self.x_paths_test) * 1.0
        correct = 0.0
        i = 1
        for path in self.x_paths_test:
            print("%.2f percent done. Accuracy: %.2f percent" % ((i / total) * 100, ((correct / i) * 100)))
            i += 1
            x = get_x(path)
            c_0, p_0 = self.internal_model.run(x)
            c_1, p_1 = self.f(x)
            if c_0 != c_1:
                correct += 1.0
        return correct / total

    # Loss
    def loss(self):
        print()

    # Train AML model
    def train(self):
        print()
        # n_runs = 1000
        # for file_path in self.x_paths_train:
        #     x = Image.open(file_path)
        #     for n in range(n_runs):

    # Test model on another dataset
    def test(self):
        print()

    # Run model on image
    # compare actual output from google model with ours
    def run(self, path):
        x = get_x(path)
        c_0, p_0 = self.internal_model.run(x)
        c_1, p_1 = self.f(x)
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


model = AML(retrieve_train_paths(), retrieve_test_paths(), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 0.05)


def run_personal(path):
    c_true, p_true, c_adv, p_adv = model.run(path)
    labels = load_labels()
    print(
        "Original: class '" + labels[str(c_true)] + "' with " + "{:.3f}".format(p_true.item() * 100) + "% probability")
    print(
        "After AML: class '" + labels[str(c_adv)] + "' with " + "{:.3f}".format(p_adv.item() * 100) + "% probability")


# x = get_x("images/imagenet/test/images/test_114.JPEG")
# new_img, _, c, p = model.f(x, 0)
# print()
run_personal("images/personal/dog.jpg")

# print(model.accuracy())
