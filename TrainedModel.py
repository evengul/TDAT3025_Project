import torchvision.models as models
import torch


class TrainedModel:
    def __init__(self):
        self.model = models.alexnet(pretrained=True)
        self.model.eval()
        self.filename = ""
        if torch.cuda.is_available():
            self.model.to('cuda')

    def run(self, x):
        input_batch = x.unsqueeze(0)  # create a mini-batch as expected by the model

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
