### In this code I apply the tools of Keras and scikit-learn machine learning libraries to address a binary classification problem. 

The data was provided by the Udemy's "Deep Learning with Python and Keras" online course by Data Weekends.

The program is aimed at classifying banknotes as either fake (class 1) or authentic (class 0) based on four numeric features. 

The main purpose of this project is to show how one can tune different hyper-parameters of a fully connected neural network  (such as the learning rate of the optimizer or the batch size at the stage of model fitting) to achieve the best performance. Performance metrics, such as the accuracy and the value of the loss function, will be plotted as functions of the number of iterations to observe the convergence speed for different values of hyper-parameters.

Moreover, I will demonstrate the use of Keras Functional API by visualizing the network's internal layers (which can be used for better understanding of the network's internal activations as well as for dimension reduction of the initial dataset).


#### Requirements

Python == 3.5

Tensorflow == 2.0

keras == 2.3.1