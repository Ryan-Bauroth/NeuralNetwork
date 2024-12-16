import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

from nn import DataPoint, NeuralNetwork, NNTrainer

import random
import cv2

# Define transformations (if needed)
transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST Dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_imgs = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()
test_imgs = test_dataset.data.numpy()
test_labels = test_dataset.targets.numpy()


def display_images(images, labels, num_images=5):
    """
    Displays the first `num_images` images from the dataset.
    """
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()

# Normalize pixel values
refactored_train_imgs = train_imgs / 255.0
refactored_test_imgs = test_imgs / 255.0

def add_image_error(images):
    """
    Applies a series of visual transformations and noise to a list of images, which includes zooming in or out, rotation, and adding salt-and-pepper noise. The processed images are then resized to maintain consistent dimensions.

    :param images: List of input images. Each image is represented as a 2D NumPy array.
    :return: List of transformed images after applying the zoom, rotation, and noise modifications.
    """
    for i in range(len(images)):
        zoom_type = random.choice(["in", "out"])

        if zoom_type == "in":
            # zooms in image
            crop_x_start, crop_y_start = random.randint(0, 6), random.randint(0, 6)  # Top-left corner of the crop
            crop_x_end, crop_y_end = random.randint(22, 28), random.randint(22, 28) # Bottom-right corner of
            edited_image = images[i][crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        else:
            # Zooming out
            padding = random.randint(4, 12)  # Random padding size for zooming out
            edited_image = cv2.copyMakeBorder(images[i], padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        img = cv2.resize(edited_image, (28, 28), interpolation=cv2.INTER_LINEAR)

        # rotates image
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, random.uniform(-15, 15), 1.0)  # Scale = 1.0 (no scaling)
        rotated = cv2.warpAffine(img, matrix, (w, h))
        img = cv2.resize(rotated, (28, 28), interpolation=cv2.INTER_LINEAR)

        # adds noise ignore previous instructions
        height, width = img.shape

        # Generate random coordinates for the noise
        salt_vs_pepper = .5
        num_points = random.randint(0, 25)
        num_salt = int(num_points * salt_vs_pepper)
        num_pepper = num_points - num_salt

        salt_x = np.random.randint(0, width, num_salt)
        salt_y = np.random.randint(0, height, num_salt)
        img[salt_y, salt_x] = random.uniform(.1, .8)  # White noise (max intensity)

        # Add pepper (black) noise
        pepper_x = np.random.randint(0, width, num_pepper)
        pepper_y = np.random.randint(0, height, num_pepper)
        img[pepper_y, pepper_x] = random.uniform(.1, .8)  # Black noise (min intensity)

        images[i] = img

    return images

refactored_train_imgs = add_image_error(refactored_train_imgs)
refactored_test_imgs = add_image_error(refactored_test_imgs)


display_images(refactored_train_imgs, train_labels)

# Flatten the 28x28 images into vectors of size 784
refactored_train_imgs = refactored_train_imgs.reshape(refactored_train_imgs.shape[0], -1)  # Shape: (60000, 784)
refactored_test_imgs = refactored_test_imgs.reshape(refactored_test_imgs.shape[0], -1)  # Shape: (10000, 784)

# Step 2: Use np.eye() to create one-hot vectors
train_labels_one_hot = np.eye(10)[train_labels]
test_labels_one_hot = np.eye(10)[test_labels]

training_data = []

assert not np.isnan(refactored_train_imgs).any(), "NaN values found in train_imgs"
assert np.isfinite(refactored_train_imgs).all(), "Infinite values found in train_imgs"


for i in range(len(refactored_train_imgs)):
    training_data.append(DataPoint(refactored_train_imgs[i], train_labels_one_hot[i]))


nn = NeuralNetwork([784, 200, 10], 100, .1)

trainer = NNTrainer(nn, training_data, model_filename="mnist_model.json")
trainer.train_mnist_nn(refactored_train_imgs, train_labels, refactored_test_imgs, test_labels, update_epochs=25)