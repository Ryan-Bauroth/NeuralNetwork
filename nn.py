import json

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import math

class DataPoint:
    def __init__(self, inputs, expected_outputs):
        """
        :param inputs: The input data or variables required for the desired operation or functionality.
        :type inputs: any

        :param expected_outputs: The expected results or outcomes associated with the given inputs.
        :type expected_outputs: any
        """
        self.inputs = inputs
        self.expected_outputs = expected_outputs


class Layer:
    def __init__(self, nodes_in, nodes_out):
        """
        :param nodes_in: Number of input nodes in the layer
        :param nodes_out: Number of output nodes in the layer
        """
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        # Corrected initialization of arrays
        self.weights = self.xavier_init((nodes_in, nodes_out))
        self.cost_gradient_w = np.zeros((nodes_in, nodes_out))
        self.biases = np.zeros(nodes_out)
        self.cost_gradient_b = np.zeros(nodes_out)

        self.inputs = None
        self.weighted_inputs = []
        self.activations = []

    def apply_gradients(self, learn_rate):
        """
        :param learn_rate: The learning rate used to scale the gradients during the update of weights and biases.
        :return: None
        """
        self.biases -= self.cost_gradient_b * learn_rate
        self.weights -= self.cost_gradient_w * learn_rate

    def clear_gradients(self):
        """
        Clears the gradients of weights and biases used in the optimization process.

        :return: None
        """
        self.cost_gradient_w = np.zeros((self.nodes_in, self.nodes_out))
        self.cost_gradient_b = np.zeros(self.nodes_out)

    def init_random_weights(self):
        """
        Initializes the weights of the neural network with random values.

        The weights are randomly sampled from a uniform distribution between -1 and 1. The resulting values
        are then scaled by the square root of the number of input nodes to normalize the weight distribution.

        :return: None
        """
        self.weights = np.random.uniform(-1, 1, (self.nodes_in, self.nodes_out)) / np.sqrt(self.nodes_in)

    def calculate_outputs(self, inputs):
        """
        :param inputs: The input data, typically a NumPy array representing the feature set to be processed by the neural network layer.
        :return: The activations resulting from applying the layer's weights, biases, and activation function to the input data.
        """
        self.inputs = inputs  # Store inputs
        # Calculate weighted inputs via vectorized dot product: inputs @ weights + biases
        self.weighted_inputs = np.dot(inputs, self.weights) + self.biases
        self.activations = self.activation_function(self.weighted_inputs)  # Apply activation function
        return self.activations

    def activation_function(self, weighted_input):
        """
        :param weighted_input: The input value or array of values to which the activation function is applied.
        :return: The output after applying the selected activation function to the input.
        """
        # Relu function
        # return max(0, weighted_input)
        # leaky relu
        return np.where(weighted_input > 0, weighted_input, .1 * weighted_input)
        # tanh function
        # return (math.exp(weighted_input) - math.exp(-weighted_input)) / (math.exp(weighted_input) + math.exp(-weighted_input))
        # Sigmoid function
        # return 1 / (1 + np.exp(-weighted_input))

    def activation_function_derivative(self, weighted_input):
        """
        :param weighted_input: The weighted input to the activation function, typically computed as a dot product of weights and inputs plus a bias term.
        :return: The derivative of the activation function evaluated at the provided weighted input. This derivative is 1 for positive values of the weighted input and 0.1 for non-positive values.
        """
        return np.where(weighted_input > 0, 1, .1)
        # activation = self.activation_function(weighted_input)
        # return 1 - activation ** 2
        # return activation * (1 - activation)

    def node_cost(self, output_activation, expected_activation):
        """
        :param output_activation: The actual output value produced by the node.
        :param expected_activation: The expected or target value for the node's output.
        :return: The computed cost for the node representing the squared difference between output_activation and expected_activation. If an OverflowError occurs during computation, returns a maximum integer value.
        """
        error = output_activation - expected_activation
        try:
            output = error  * error
        except OverflowError:
            output = np.iinfo(np.int32).max
        return output

    def node_cost_derivative(self, output_activation, expected_activation):
        """
        :param output_activation: The activation value output from the node.
        :param expected_activation: The desired or expected activation value for the node.
        :return: The derivative of the cost function with respect to the node's output activation.
        """
        return 2 * (output_activation - expected_activation)

    def xavier_init(self, shape):
        """
        :param shape: A tuple representing the dimensions of the weight matrix to be initialized. The first element is the number of input units, and the second is the number of output units.
        :return: A NumPy array of weights initialized using the Xavier initialization method with uniform distribution.
        """
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape)

    def calculate_output_layer_node_values(self, expected_outputs):
        """
        :param expected_outputs: A list of expected output values for the output layer nodes. It represents the target values used to calculate the error for each node.
        :return: A list of node values for the output layer, calculated as the product of the cost derivative and the activation function derivative for each node.
        """
        node_values = []

        for node_out in range(len(expected_outputs)):
            cost_derivative = self.node_cost_derivative(self.activations[node_out], expected_outputs[node_out])
            activation_derivative = self.activation_function_derivative(self.weighted_inputs[node_out])
            node_values.append(activation_derivative * cost_derivative)

        return node_values

    def calculate_hidden_layer_node_values(self, old_layer, old_node_values):
        """
        :param old_layer: The previous layer object containing weights used in the calculation
        :param old_node_values: A numpy array representing the node values of the previous layer
        :return: A numpy array representing the computed node values for the current hidden layer
        """
        # Vectorized computation of new node values
        weighted_input_derivative = np.dot(old_layer.weights, old_node_values)
        activation_derivative = self.activation_function_derivative(self.weighted_inputs)

        # New node values calculated by element-wise multiplication
        new_node_values = weighted_input_derivative * activation_derivative
        return new_node_values


    def update_gradients(self, node_values):
        """
        :param node_values: The computed gradients of the output nodes with respect to the cost function.
        :return: Updates the internal cost gradients for weights and biases by adding the computed values based on the provided node gradients.
        """
        # Vectorized calculation of the gradients for weights (W) and biases (B)

        # Gradient with respect to weights (W): input values * node values
        self.cost_gradient_w += np.outer(self.inputs, node_values)

        # Gradient with respect to biases (B): directly node values
        self.cost_gradient_b += node_values



class NeuralNetwork:
    def __init__(self, layer_sizes, batch_size, learn_rate=.01, h=.0001):
        """
        :param layer_sizes: A list of integers representing the number of neurons in each layer of the neural network. The length of the list determines the number of layers in the network.
        :param batch_size: An integer representing the number of training examples per batch for gradient descent.
        :param learn_rate: A float representing the learning rate used in optimization, which controls the step size during weight updates.
        :param h: A small float value used for numerical stability in certain operations or regularization.
        """
        self.layers = []
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.h = h
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i + 1])
            layer.init_random_weights()
            self.layers.append(layer)

    def learn(self, training_batch):
        """
        Trains the model using the provided training batch. This function processes each data point in the batch to update gradients and then applies these accumulated gradients to the model based on a learning rate.

        :param training_batch: A collection of data points used for training. Each data point is processed to update the model's gradients.
        :return: None
        """
        # Accumulate gradients for the batch
        for data_point in training_batch:
            self.update_all_gradients(data_point)

        # Apply the accumulated gradients
        self.apply_all_gradients(self.learn_rate / len(training_batch))

        self.clear_all_gradients()

    def train(self, training_data):
        """
        :param training_data: List of training data that needs to be processed and trained on.
        :return: None
        """
        indices = np.random.permutation(len(training_data))
        shuffled_training_data = [training_data[i] for i in indices]

        for i in range(0, len(shuffled_training_data), self.batch_size):
            batch_data = shuffled_training_data[i:i + self.batch_size]

            if len(batch_data) == 0:
                break

            self.learn(batch_data)

    def clear_all_gradients(self):
        """
        Clears the gradients of all layers in the model. This is typically used during the training process to reset the gradient values after a weight update step.

        :return: None
        """
        for layer in self.layers:
            layer.clear_gradients()

    def old_learn(self, training_data):
        """
        Deprecated h => 0 based learning function

        :param training_data: The dataset used for training the model. It is expected to be a collection of samples from which the model will update its parameters.
        :return: None
        """
        indices = np.random.permutation(len(training_data))

        shuffled_training_data = [training_data[i] for i in indices]

        for i in range(0, len(shuffled_training_data), self.batch_size):
            training_data = shuffled_training_data[i:i + self.batch_size]

            if len(training_data) == 0:
                break

            for layer in self.layers:
                for node_in in range(layer.nodes_in):
                    for node_out in range(layer.nodes_out):
                        original_cost = self.cost(training_data)
                        layer.weights[node_in][node_out] += self.h
                        new_cost = self.cost(training_data)
                        delta_cost = new_cost - original_cost
                        layer.cost_gradient_w[node_in][node_out] = delta_cost / self.h
                        layer.weights[node_in][node_out] -= self.h

                for bias_ind in range(layer.nodes_out):
                    original_cost = self.cost(training_data)
                    layer.biases[bias_ind] += self.h
                    new_cost = self.cost(training_data)
                    delta_cost = new_cost - original_cost
                    layer.cost_gradient_b[bias_ind] = delta_cost / self.h
                    layer.biases[bias_ind] -= self.h

            self.apply_all_gradients(self.learn_rate)

    def update_all_gradients(self, data_point):
        """
        updates all gradients with backpropagation algorithm
        :param data_point: An object representing a single data point consisting of input values and expected output values.
        :return: None
        """
        # backpropagation algorithm
        self.calculate_outputs(data_point.inputs)

        output_layer = self.layers[-1]
        node_values = output_layer.calculate_output_layer_node_values(data_point.expected_outputs)
        output_layer.update_gradients(node_values)

        for hidden_layer_index in range(len(self.layers) - 2, -1, -1):
            hidden_layer = self.layers[hidden_layer_index]
            node_values = hidden_layer.calculate_hidden_layer_node_values(self.layers[hidden_layer_index + 1], node_values)
            hidden_layer.update_gradients(node_values)

    def apply_all_gradients(self, learn_rate):
        """
        Applies gradients to each layer

        :param learn_rate: Learning rate to be applied when updating the gradients in each layer.
        :return: None
        """
        for layer in self.layers:
            layer.apply_gradients(learn_rate)

    def calculate_outputs(self, inputs):
        """
        Calculates the layer outputs

        :param inputs: The input data to be processed by the layers of the network. Typically, this is a list, array, or another iterable structure containing numerical values.
        :return: The transformed output data after sequentially processing through all layers of the network.
        """
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def classify(self, inputs):
        """
        Returns outputs of final function

        :param inputs: Input data to be classified. The exact structure and type depend on the implementation requirements.
        :return: The classification result based on the provided inputs.
        """
        outputs = self.calculate_outputs(inputs)
        return outputs

    def cost(self, training_data):
        """
        Calculates the average cost of all training points

        :param training_data: A collection of data points used to calculate the cost. Each data point should contain inputs and expected outputs.
        :return: The average cost computed over all provided training data points.
        """
        total_cost = 0
        for data_point in training_data:
            outputs = self.calculate_outputs(data_point.inputs)
            output_layer = self.layers[-1]
            for node_out in range(len(outputs)):
                total_cost += output_layer.node_cost(outputs[node_out], data_point.expected_outputs[node_out])
        return total_cost / len(training_data)

    def save_model(self, file_path):
        """
        Saves model

        :param file_path: The path to the file where the model data will be saved.
        :return: None
        """
        model_data = {
            "weights": [],
            "biases": []
        }
        for layer in self.layers:
            model_data["weights"].append(layer.weights.tolist())
            model_data["biases"].append(layer.biases.tolist())

        with open(file_path, "w") as f:
            json.dump(model_data, f)


    def load_model(self, file_path):
        """
        Loads model

        :param file_path: The path to the JSON file containing model weights and biases.
        :return: None
        """
        # Load weights and biases from JSON
        with open(file_path, "r") as f:
            model_data = json.load(f)
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].weights = np.array(model_data["weights"][layer_idx])
            self.layers[layer_idx].biases = np.array(model_data["biases"][layer_idx])


class NNTrainer:
    def __init__(self, nn, training_data, model_filename="nn_model.json"):
        """
        :param nn: Represents the neural network to be used in the implementation.
        :param training_data: The dataset used to train the neural network.
        :param model_filename: Optional parameter specifying the filename for saving or loading the neural network model. Defaults to "nn_model.json".
        """
        self.nn = nn
        self.training_data = training_data
        self.model_filename = model_filename

    def generate_sample_inputs(self, random_point_num=500):
        """
        Generates sample inputs based upon a sample function (for example: y > sin(6x))

        :param random_point_num: The number of random sample points to generate for input data.
        :return: None. The function populates the training_data attribute with generated data points and their corresponding expected results.
        """
        sample_inputs = np.random.rand(random_point_num, 2)

        sample_expected_result = [self.gradient_equation(x) for x in sample_inputs]

        sample_training_data = [
            DataPoint(sample_inputs[i], [sample_expected_result[i], abs(1 - sample_expected_result[i])]) for i in
            range(len(sample_inputs))]

        self.training_data = sample_training_data

    def gradient_equation(self, coords):
        """
        The sample equation for generating sample_inputs

        :param coords: A tuple containing two float values representing coordinates (x, y).
        :return: Returns 1 if the sine of (x * 6) is greater than y; otherwise, returns 0.
        """
        x, y = coords
        # return 1 if x > .5 and y > .5 else 0
        return 1 if math.sin(x * 6) > y else 0
        # return 1 if x + (y - .2) * (y - .2) > .5 else 0

    def train_function_based_nn(self, to_completion=True, epochs=1000, update_epochs=500):
        """
        Trains the model based upon the gradient_equation generated data

        :param to_completion: A boolean flag that determines whether the training should continue until a completion condition is met.
        :param epochs: An integer specifying the total number of epochs for training when `to_completion` is False.
        :param update_epochs: An integer defining the number of epochs after which the model's progress is evaluated and updated.
        :return: None
        """
        training_data_inputs = np.array([data_point.inputs for data_point in self.training_data])
        training_data_expected_outputs = np.array([data_point.expected_outputs for data_point in self.training_data])

        if to_completion:
            done_training = False

            epochs = 0

            costs = []

            while not done_training:
                training_costs = self.train_model(update_epochs, cost_return=True)
                epochs += update_epochs
                costs.extend(training_costs)
                rounded_sample_result = [1 if self.nn.calculate_outputs(x)[0] > .5 else 0 for x in training_data_inputs]
                done_training = self.twod_plots(training_data_inputs, training_data_expected_outputs, rounded_sample_result)
                self.cost_plot(costs)
                self.nn.save_model(self.model_filename)

            print(epochs)
        else:
            self.train_model(epochs)

    def train_mnist_nn(self, train_images, train_labels, test_images, test_labels, to_completion=True, epochs=1000, update_epochs=500):
        """
        Trains the model based upon the mnist data

        :param train_images: The training dataset containing images for training the neural network.
        :param train_labels: The corresponding labels for the training images.
        :param test_images: The testing dataset containing images for evaluating the neural network's performance.
        :param test_labels: The corresponding labels for the testing images.
        :param to_completion: A boolean flag that determines whether training should continue until a predefined accuracy threshold is reached. Default is True.
        :param epochs: The number of training epochs to perform when `to_completion` is set to False. Default is 1000.
        :param update_epochs: The number of epochs after which performance metrics are updated and checkpoints are saved when `to_completion` is True. Default is 500.
        :return: None
        """
        if to_completion:
            done_training = False

            epochs = 0

            costs = []
            mnist_training_correct_percent = [.1]
            mnist_testing_correct_percent= [.1]

            while not done_training:
                training_costs = self.train_model(update_epochs, cost_return=True)
                epochs += update_epochs
                costs.extend(training_costs)
                self.cost_plot(costs)
                self.nn.save_model(self.model_filename)
                train_correct_percent, test_correct_percent = self.test_mnist_model(train_images, train_labels, test_images, test_labels)
                mnist_training_correct_percent.append(train_correct_percent)
                mnist_testing_correct_percent.append(test_correct_percent)
                done_training = train_correct_percent > .99 and test_correct_percent > .99
                self.accuracy_percent_plot([mnist_training_correct_percent, mnist_testing_correct_percent], title="NN Training Accuracy Line Plot")

            print(epochs)
        else:
            self.train_model(epochs)

    def test_mnist_model(self, train_images, train_labels, test_images, test_labels):
        """
        Generates test images for the mnist model

        :param train_images: A list or array of training images used for evaluating the model.
        :param train_labels: A list or array of corresponding labels for the training images.
        :param test_images: A list or array of testing images used for evaluating the model.
        :param test_labels: A list or array of corresponding labels for the testing images.
        :return: A tuple containing the training accuracy and testing accuracy as floats.
        """
        train_correct = 0
        test_correct = 0
        for i in range(len(train_images)):
            classification = np.argmax(self.nn.classify(train_images[i]))
            actual_classification = train_labels[i]
            if classification == actual_classification: train_correct += 1
        for i in range(len(test_images)):
            classification = np.argmax(self.nn.classify(test_images[i]))
            actual_classification = test_labels[i]
            if classification == actual_classification: test_correct += 1
        return train_correct / len(train_images), test_correct / len(test_images)

    def cost_plot(self, costs):
        """
        Plots a cost vs iterations/epochs graph

        :param costs: List or array containing the cost values to be plotted, where each element represents the cost at a specific iteration.
        :return: None. The function directly generates and displays a line plot showing the progression of costs over iterations/epochs.
        """
        # plots a cost line chart, showing how cost decreases over iterations
        self.plot_line([np.arange(len(costs)), costs], title="NN Cost Line Plot", x="Epochs", y="Cost")

    def accuracy_percent_plot(self, accuracy_percent_arrs, title="NN Accuracy Line Plot"):
        """
        Plots two lines comparing test and train accuracy %

        :param accuracy_percent_arrs: A list containing arrays of accuracy percentages for training clumps. Each array represents a separate set of accuracy data to be plotted.
        :param title: The title of the plot. Default is "NN Accuracy Line Plot".
        :return: None. Displays a line plot visualizing the accuracy percentages for the given training clumps.
        """
        self.plot_lines([np.arange(len(accuracy_percent_arrs[0])), accuracy_percent_arrs[0]],[np.arange(len(accuracy_percent_arrs[1])), accuracy_percent_arrs[1]], title=title, x="Training Clumps", y="Accuracy")


    def twod_plots(self, sample_inputs, expected_result, result):
        """
        Plots 2D test scatters (actual output, rounded output, and green/red output)

        :param sample_inputs: Array of input data points for which neural network predictions and comparisons are made.
        :param expected_result: Array of expected true outputs corresponding to the sample inputs.
        :param result: Array of actual neural network outputs after rounding or simplification.
        :return: Boolean indicating whether all points have been correctly identified (all points green in the red/green scatter plot).
        """
        # array of the actual nn predictions
        colormap_colors = [self.nn.calculate_outputs(x)[0] for x in sample_inputs]
        # array of colors highlighting if each point is correctly marked
        rg_colors = ["green" if np.array_equal(expected_result[i][0], result[i]) else "red" for i in range(len(expected_result))]
        # condition checking whether all points are correctly identified
        all_green = np.all(rg_colors == 'green')

        # color map and simplified inputs array
        cmap = plt.get_cmap('viridis')
        scatter_inputs = [sample_inputs[:, 0], sample_inputs[:, 1]]

        # plots all 3 scatter plots using different inputs and information
        # actual nn predictions scatter plot
        self.plot_scatter(scatter_inputs, colormap_colors, title="Actual Output Scatter Plot", color_map=cmap, figure_size=(13, 10))
        # rounds true/false values from nn predictions and displays them
        self.plot_scatter(scatter_inputs, result, title="Rounded Output Scatter Plot")
        # displays which points are correctly identified
        self.plot_scatter(scatter_inputs, rg_colors, title="Red/Green Scatter Plot")
        
        return all_green

    def plot_scatter(self, scatter_inputs, c, title="Scatter Plot", figure_size=(10, 10), s=50, x="X", y="Y", color_map=None):
        """
        Plots a single scatter

        :param scatter_inputs: A tuple containing two arrays or lists representing the x and y coordinates of the points to be plotted.
        :param c: The color or sequence of colors to be used for the points. Can be a single color or an array of colors matching the number of points.
        :param title: The title of the scatter plot. Default is "Scatter Plot".
        :param figure_size: The size of the figure in inches as a tuple (width, height). Default is (10, 10).
        :param s: The size of the markers in the scatter plot. Default is 50.
        :param x: Label for the x-axis. Default is "X".
        :param y: Label for the y-axis. Default is "Y".
        :param color_map"""
        # sets size of figure
        plt.figure(figsize=figure_size)
        
        # scatters input points with color and size
        plt.scatter(scatter_inputs[0], scatter_inputs[1], c=c, s=s)
        
        # if color map is included, shows a color bar and adds the color map
        if color_map is not None:
            plt.scatter(scatter_inputs[0], scatter_inputs[1], vmin=0, vmax=1, c=c, cmap=color_map, s=s)
            plt.colorbar()
        # if not, plots normal color input
        else:
            plt.scatter(scatter_inputs[0], scatter_inputs[1], c=c, s=s)
            
        # sets title and axis labels
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        
        # adds grid and shows graph
        plt.grid(True)
        plt.show()
        
    def plot_line(self, inputs, title="Line Plot", figure_size=(10, 6), x="X", y="Y", color='b', linestyle='-', linewidth=2):
        """
        Plots a single line

        :param inputs: A tuple of two lists or arrays, representing the x and y data for the line plot.
        :param title: The title of the plot. Default is "Line Plot".
        :param figure_size: A tuple representing the size of the figure (width, height). Default is (10, 6).
        :param x: Label for the x-axis. Default is "X".
        :param y: Label for the y-axis. Default is "Y".
        :param color: Color of the plotted line. Default is 'b' (blue).
        :param linestyle: Style of the line (e.g., '-', '--', '-.'). Default is '-'.
        :param linewidth: Width of the line. Default is 2.
        :return: Displays the line plot.
        """
        # sets size of figure
        plt.figure(figsize=figure_size)
        
        # plots line with two input arrays, a color, a linestyle, and a line width
        plt.plot(inputs[0], inputs[1], color=color, linestyle=linestyle, linewidth=linewidth)
        
        # sets title and axis titles
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

        plt.ylim(0, 1)  # Set x-axis range from 0 to 6
        
        # adds grid and shows graph
        plt.grid(True)
        plt.show()

    def plot_lines(self, inputs_one, inputs_two, title="Line Plot", figure_size=(10, 6), x="X", y="Y", colors=['g', 'r'], linestyle='-',
                  linewidth=2):
        """
        Plots two lines

        :param inputs_one: Tuple of two arrays representing x and y values for the first line.
        :param inputs_two: Tuple of two arrays representing x and y values for the second line.
        :param title: Title of the plot as a string. Default is "Line Plot".
        :param figure_size: Tuple specifying the width and height of the plot figure. Default is (10, 6).
        :param x: Label for the x-axis. Default is "X".
        :param y: Label for the y-axis. Default is "Y".
        :param colors: List of colors for the plot lines. Default is ['g', 'r'].
        :param linestyle: Line style for the plot lines. Default is '-'.
        :param linewidth: Line width for the plot lines. Default is 2.
        :return: None
        """
        # sets size of figure
        plt.figure(figsize=figure_size)

        # plots line with two input arrays, a color, a linestyle, and a line width
        plt.plot(inputs_one[0], inputs_one[1], color=colors[0], linestyle=linestyle, linewidth=linewidth)
        plt.plot(inputs_two[0], inputs_two[1], color=colors[1], linestyle=linestyle, linewidth=linewidth)

        # sets title and axis titles
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

        plt.ylim(0, 1)  # Set x-axis range from 0 to 6

        # adds grid and shows graph
        plt.grid(True)
        plt.show()

    def train_model(self, epochs, cost_return=False):
        """
        Trains the model for a given # of epochs

        :param epochs: Number of training iterations to be executed on the neural network.
        :param cost_return: Boolean flag to indicate whether to collect and return training costs for each iteration.
        :return: A list of costs at each iteration if cost_return is True, otherwise an empty list.
        """
        # Adjust learning iterations
        costs = []
        for _ in tqdm(range(epochs)):
            self.nn.train(self.training_data)
            if cost_return:
                costs.append(self.nn.cost(self.training_data))
        return costs


if __name__ == "__main__":
    # Initialize network
    h = 0.0001
    learn_rate = 0.01
    nn = NeuralNetwork([2, 5, 2], 64, learn_rate=learn_rate, h=h)  # 2 inputs, 3 hidden nodes, 2 outputs

    trainer = NNTrainer(nn, None)

    trainer.generate_sample_inputs()
    trainer.train_function_based_nn(to_completion=True, update_epochs=500)