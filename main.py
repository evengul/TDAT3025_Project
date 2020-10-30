import torch
import urllib
from PIL import Image
from torchvision import transforms
import torchvision.models as models

device = 'cpu'
if torch.cuda.is_available():
    device = 'gpu:0'
print("Using device: " + str(device))


class TrainedModel:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
        self.model.eval()
        self.filename = ""
        if torch.cuda.is_available():
            self.model.to('cuda')

    def run(self, personal, isMatrix, imageName="", matrix=torch.zeros((224, 224, 3))):
        if personal:
            self.filename = "images/personal/" + imageName
        else:
            self.filename = "images/imageNet/" + imageName

        input_image = matrix
        if not isMatrix:
            input_image = Image.open(self.filename)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.

        # Returns probabilities for each class
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        classification = torch.argmax(probabilities).item()
        return classification, probabilities[classification]


class AML:
    def __init__(self, x_train, y_train, x_test, y_test, Q, learning_rate):
        self.internal_model = TrainedModel()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.Q = Q
        self.learning_rate = learning_rate

    # Predictor
    def f(self):
        print()

    # Accuracy
    def accuracy(self):
        print()

    # Loss
    def loss(self):
        print()

    # Train AML model
    def train(self):
        print()

    # Test model on another dataset
    def test(self):
        print()

    # Run model on image
    # compare actual output from google model with ours
    def run(self):
        print()


model = AML(None, None, None, None, None, 0.1)
print(model.internal_model.run(True, False, "dog.jpg"))
