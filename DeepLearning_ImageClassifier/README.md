### An image classifier aimed at recognizing the gender of a person from pictures. Given the data, the program can be easily extended to a multi classificiation problem.

 
Because of a large number of parameters to be trained, I suggest you run this on a cloud with a GPU.

Download and unpack the male/female pictures from https://www.dropbox.com/s/nov493om2jmh2gp/male_female.tgz?dl=0. These images and labels were obtained from Crowdflower. 

I am not uploading the files to GitHub because of the size, but after the download one should end up having a folder 'data' with two subfolder 'train' and 'test', each of which in turn contains two subfolders '0_female' and '1_male'. 

We will follow the steps below:

- have a look at the directory structure and inspect a couple of pictures
- design a model that will take a color image of size 64x64 as input and return a binary output (female=0/male=1)
- introduce some kind of regularization technique in the model (Dropout, Batch Normalization, etc.)
- choose the optimizer and compile the model 
- using ImageDataGenerator, define a train generator that will augment the images with some geometric transformations.
- define also a test generator, whose only purpose is to rescale the pixels by 1./255
- use the function flow_from_directory to generate batches from the train and test folders. Set the target_size to 64x64.
- Use the model.fit_generator function to fit the model on the batches generated from the ImageDataGenerator. Since we are streaming and augmenting the data in real time, we need to decide how many batches make an epoch and how many epochs to run
- train the model. Evaluate accuracy on the test data utilizing the AUROC metric.
- after the training, check a few of the misclassified pictures. Are those sensible errors?  

#### Requirements

Python == 3.5

Tensorflow == 2.0

keras == 2.3.1