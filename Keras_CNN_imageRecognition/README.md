### In this code I apply Keras API for an image recognition task. The solution employs a Convolutional Neural Network architecture.

We will work with the Cifar 10 Dataset, a very famous dataset that contains images for 10 different categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Here's what we will do:

- load the cifar10 dataset using `keras.datasets.cifar10.load_data()`
- display a few images, see how hard/easy it is to recognize an object with such low resolution
- check the shape of X_train, does it need reshape?
- check the scale of X_train, does it need rescaling?
- check the shape of y_train, does it need reshape?
- build a model with the following architecture, and choose the parameters and activation functions for each of the layers:
    - conv2d
    - conv2d
    - maxpool
    - conv2d
    - conv2d
    - maxpool
    - flatten
    - dense
    - output
- compile the model and check the number of parameters
- attempt to train the model with the optimizer of our choice. How fast does training proceed?

Note: because the number of trainable parameters is expected to be quite large, it is preferable to work in a cloud with the help of GPU acceleration. 

#### Requirements

Python == 3.5

Tensorflow == 2.0

keras == 2.3.1