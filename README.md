# dog_breed_classifier
As a part of this hands on project for the Udacity deep learning nanodegree course, I created a CNN (Convolutional neural network) from scratch. This algorithm does the following:
1. identifies a dog picture with it's breed
2. identifies a human face and predicts the closest dog breed
3. displays an error when the image is neither dog or human

# Datasets for this project can be downloaded here:
1. Download the https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
2. Download the human_dataset https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip

# CNN architecture used for classifying dog breeds: 
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=6272, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=133, bias=True)
  (dropout): Dropout(p=0.2)
)
Hyperparameters for training : Learning rate = 0.03, epochs = 50
Accuracy obtained on test : 14% (baseline : 10%)

# Transfer learning (modified ResNet to obtain output for 133 classes):
Hyperparameters for training : Learning rate = 0.005, epochs = 25
Accuracy obtained on test : 81%
