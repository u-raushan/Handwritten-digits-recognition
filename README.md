# Handwritten-digit-recognition
This is part of the Machine Learning course by Coursera. The repository contains logistic regression and neural network models for handwritten digit recognition written in Matlab. The output of the models is 10 classes representing digits from '0' to '9'.

data.mat file contains 5,000 training examples. Random 100 data points are displayed below.

<img width="276" alt="image" src="https://user-images.githubusercontent.com/69568898/191809509-81e42c68-1555-4831-b4c8-09d8ee9e1a03.png">

1. Classification using logistic regression with regularization (one-vs-all)

A cost function is as follows:

<img width="409.2" alt="image" src="https://user-images.githubusercontent.com/69568898/191832632-dfadeab9-c81f-4590-b4dc-36644f0c6247.png">

To minimize the cost function, theta values are updated as follows:

<img width="325.2" alt="image" src="https://user-images.githubusercontent.com/69568898/191833086-99813eb7-5dfb-4aa7-a3a3-02c268abc2ca.png">


2. Classification using a pretrained neural network (NN)

NN consists of three layers: input, hidden, and output layers. 

The picture size of a handwritten digit is 20 by 20 pixels so the input layer size is 400 units. The hidden layer size is 25 units (can be any value). The output layer size is 10 units (10 digits).

The model is pretrained. Theta 1 and theta 2 parameters are part of the course materials.
