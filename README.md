# TDAT3025_Project
This project is the final result of Team 2's work in TDAT3025, NTNU. 

The model contained in the code is a model to make GoogLeNet misclassify images by generating a noise layer to add to the images before they are run through the internal model.

To run the program, the Tiny ImageNet dataset needs to be downloaded [from here](http://www.image-net.org/image/tiny/tiny-imagenet-200.zip), and put in the "images/imagenet/" folder.
Then you can run `python3 main.py`which will run the model with the functionality as it stands on delivery. Changeable/Testable hyperparameters of the model is the learning rate
and how many tiles the noise is split into. Higher tile level will drastically increase runtime.

It is also possible to run personal images with the function `run_personal()` to try and misclassify these.

Note: The code has been run on a NVIDIA GPU and will be quite slow an a CPU.
