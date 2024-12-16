import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

from nn import NeuralNetwork, DataPoint

# Define transformations (if needed)
transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST Dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_imgs = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()
test_imgs = test_dataset.data.numpy()
test_labels = test_dataset.targets.numpy()

# Normalize pixel values
refactored_train_imgs = train_imgs / 255.0
refactored_test_imgs = test_imgs / 255.0

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

nn = NeuralNetwork([784, 200, 10], 100, .1)
nn.load_model("mnist_model.json")

# Flatten the 28x28 images into vectors of size 784
refactored_train_imgs = refactored_train_imgs.reshape(refactored_train_imgs.shape[0], -1)  # Shape: (60000, 784)
refactored_test_imgs = refactored_test_imgs.reshape(refactored_test_imgs.shape[0], -1)  # Shape: (10000, 784)

# Step 2: Use np.eye() to create one-hot vectors
train_labels_one_hot = np.eye(10)[train_labels]
test_labels_one_hot = np.eye(10)[test_labels]

training_data = []

for i in range(len(refactored_train_imgs)):
    training_data.append(DataPoint(refactored_train_imgs[i], train_labels_one_hot[i]))

failed_images = []

for data_point in training_data:
    classification = nn.calculate_outputs(data_point.inputs)
    if np.argmax(classification) != np.argmax(data_point.expected_outputs):
        new_data_point = DataPoint(data_point.inputs, classification)
        failed_images.append(new_data_point)

failed_inputs_stack = []
failed_labels_stack = []

for data_point in failed_images:
    if len(failed_inputs_stack) < 10:
        failed_inputs_stack.append(data_point.inputs.reshape(28, 28))
        failed_labels_stack.append(np.argmax(data_point.expected_outputs))
    else:
        failed_inputs_stack.append(data_point.inputs.reshape(28, 28))
        failed_labels_stack.append(np.argmax(data_point.expected_outputs))
        display_images(failed_inputs_stack, failed_labels_stack, num_images=len(failed_inputs_stack))
        failed_inputs_stack = []
        failed_labels_stack = []
        break

print(1 - len(failed_images) / 60000)