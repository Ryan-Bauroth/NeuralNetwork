# Neural Network
## Description

[Let's talk about AI](https://open.spotify.com/track/4h5x3XHLVYFJaItKuO2rhy?si=2ddf9a4aaf864a36). Rather, let's talk about this project, which is a **neural network** built from scratch (no tensorflow, no pytorch, nothing). 

So far, I have used this neural network to successfully complete two tasks:

- Train on and find a **defined function** (for example, y < sin(6x))
- Train on a dataset of **MNIST handwritten digits** and detect handwritten digits through a pygame interface

## Installation & Use

For installation, download this GitHub repo and run any of the files!

## Functionality

### The Neural Network

The custom neural network (NN) takes a learning rate (float), training data (arr[DataPoints]), and a structure (arr). The NN then uses a backpropagation algorithm, utilizing gradient descent, in order to train.
In the NN file, there are several activation functions to choose from depending on your data (LeakyRELU is recommended as a starting point). 

### Function Training

For training on a specific mathematical function, the NN file uses the method 'gradient_function' in order to generate any number of starting training values. The NN then trains on this data, providing debug plots as shown below:

<img width="543" alt="Screenshot 2024-12-17 at 8 14 38 PM" src="https://github.com/user-attachments/assets/7cd404b2-43b6-4ecb-8fc5-0ab1a57d0059" />

_Scatter plot showing actual NN results in 2D space_

<img width="418" alt="Screenshot 2024-12-17 at 8 14 45 PM" src="https://github.com/user-attachments/assets/8d405bd1-4004-49d7-b874-4ab29047f671" />

_Scatter plot showing rounded NN results in 2D space_

<img width="416" alt="Screenshot 2024-12-17 at 8 14 51 PM" src="https://github.com/user-attachments/assets/b408f637-5bcc-4771-9dbe-53e344570b93" />

_Scatter plot showing NN accuracy (green = accurate prediction, red = inaccurate prediction)_

<img width="590" alt="Screenshot 2024-12-17 at 8 14 59 PM" src="https://github.com/user-attachments/assets/6464bbac-50dc-4cad-9ceb-f9fda4458d6e" />

_Line plot showing cost vs epochs_

As you can see, the NN very accurately found the sin(6x) function, albeit given a large number of epochs. 

### MNIST Handwriting Training

For training on the MNIST handwritten digits, the mnist_train file uses the NN and NN Trainer class in order to train from 60,000 28x28 pixel monochrome images of handwritten digits. The file an image modification algorithm randomly resizes and rotates each image, in addition to adding salt and pepper noise, in an effort to avoid overfitting.
Example handwritten digits after image modification algorithm:

<img width="589" alt="Screenshot 2024-12-17 at 8 21 36 PM" src="https://github.com/user-attachments/assets/2cbf53f4-c637-4a95-ad62-df7760fec9ee" />

_Five handwritten digits resized, rotated, and with salt and pepper noise applied_

The mnist_test file displays an accuracy rating of the model in the terminal for both training and test data. It also displays 10 example images of where the model made mistakes to help debug. Currently, my model has an accuracy of 93.05%.

<img width="997" alt="Screenshot 2024-12-17 at 8 24 42 PM" src="https://github.com/user-attachments/assets/b1ea0707-91ed-4a09-a634-356cf206dcee" />

_Ten mistakenly classified digits, labeled the model identifications_


Finally, the mnist_drawing_test file opens a pygame window to allow users to draw their own digits. These are identified by the NN in real time and a % ranking is displayed on the side for each possible classification.

<img width="496" alt="Screenshot 2024-12-17 at 8 29 56 PM" src="https://github.com/user-attachments/assets/c1ca3f3e-ce9a-4270-9610-f8c011795258" />

_A drawing of the number six with the NN identifying it with a 95% likelihood_

While this outlet is not perfect, as it does not accurately represent the training images, it is more applicable to real life and also more fun.

## Resources

Here are some of the resources I used while creating this project

- This project is based upon [this video by Sebastian Lague](https://www.youtube.com/watch?v=hfMk-kjRv4c&t=2846s)
- I used ChatGPT and JebrainsAI in a limited fashion on this project

