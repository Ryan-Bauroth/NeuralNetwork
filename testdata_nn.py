import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import math

class DataPoint:
    def __init__(self, inputs, expected_outputs):
        self.inputs = inputs
        self.expected_outputs = expected_outputs


class Layer:
    def __init__(self, nodes_in, nodes_out):
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        # Corrected initialization of arrays
        self.weights = np.zeros((nodes_in, nodes_out))
        self.cost_gradient_w = np.zeros((nodes_in, nodes_out))
        self.biases = np.zeros(nodes_out)
        self.cost_gradient_b = np.zeros(nodes_out)

    def apply_gradients(self, learn_rate):
        for node_out in range(self.nodes_out):
            self.biases[node_out] -= self.cost_gradient_b[node_out] * learn_rate
            for node_in in range(self.nodes_in):
                self.weights[node_in][node_out] -= self.cost_gradient_w[node_in][node_out] * learn_rate

    def init_random_weights(self):
        for node_in in range(self.nodes_in):
            for node_out in range(self.nodes_out):
                random_value = np.random.uniform(-1, 1)
                self.weights[node_in][node_out] = random_value / math.sqrt(self.nodes_in)

    def calculate_outputs(self, inputs):
        activations = np.zeros(self.nodes_out)
        for node_out in range(self.nodes_out):
            weighted_input = (self.biases[node_out])
            for node_in in range(self.nodes_in):
                weighted_input += inputs[node_in] * self.weights[node_in][node_out]
            activations[node_out] = self.activation_function(weighted_input)
        return activations

    def activation_function(self, weighted_input):
        # Sigmoid function
        return 1 / (1 + np.exp(-weighted_input))

    def node_cost(self, output_activation, expected_activation):
        error = output_activation - expected_activation
        return error * error


class NeuralNetwork:
    def __init__(self, layer_sizes, batch_size, learn_rate=.01, h=.0001):
        self.layers = []
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.h = h
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i + 1])
            layer.init_random_weights()
            self.layers.append(layer)

    def learn(self, training_data):

        indices = np.random.permutation(len(training_data))

        shuffled_training_data = [training_data[i] for i in indices]

        for i in range(0, len(shuffled_training_data), self.batch_size):
            training_data = shuffled_training_data[i:i + self.batch_size]

            if len(training_data) < self.batch_size:
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

            self.apply_all_gradients()

    def apply_all_gradients(self):
        for layer in self.layers:
            layer.apply_gradients(self.learn_rate)

    def calculate_outputs(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def classify(self, inputs):
        outputs = self.calculate_outputs(inputs)
        return np.argmax(outputs)  # Corrected classification mechanism

    def cost(self, training_data):
        total_cost = 0
        for data_point in training_data:
            outputs = self.calculate_outputs(data_point.inputs)
            output_layer = self.layers[-1]
            for node_out in range(len(outputs)):
                total_cost += output_layer.node_cost(outputs[node_out], data_point.expected_outputs[node_out])
        return total_cost / len(training_data)


class RunNN:
    def __init__(self, nn, training_data):
        self.nn = nn
        self.training_data = training_data

    def plot_gradient_graph(self, random_point_num=500):
        cmap = plt.get_cmap('viridis')
        plt.figure(figsize=(10, 6))

        sample_inputs = np.random.rand(random_point_num, 2)
        real_inputs = np.array([data_point.inputs for data_point in training_data])
        real_outputs = np.array([data_point.expected_outputs[0] for data_point in training_data])

        sample_colors = [self.nn.calculate_outputs(x)[0] for x in sample_inputs]
        colors = [self.nn.calculate_outputs(x)[0] for x in real_inputs]

        plt.scatter(sample_inputs[:, 0], sample_inputs[:, 1], c=sample_colors, cmap=cmap, s=50)
        plt.scatter(real_inputs[:, 0], real_inputs[:, 1], c=real_outputs, cmap=cmap, s=100)

        # Plot data
        plt.colorbar()
        plt.title('Data Plot')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.grid(True)
        plt.show()


# Load the data
data = pd.read_csv('new_data.csv')

# set inputs to data values, excluding 'P' value
inputs = data[['X', 'Y']].values
# sets the expected output for each input using the 'P' value
expected_outputs = [[1, 0] if val == 1 else [0, 1] for val in data['P']]

# Create list of data points for training
training_data = [DataPoint(inputs[i], expected_outputs[i]) for i in range(len(inputs))]

# Initialize network, learn
h = .0001
learn_rate = 0.01
nn = NeuralNetwork([2, 3, 2], 8, learn_rate=learn_rate, h=h)  # 2 inputs, 3 hidden nodes, 2 outputs

# Adjust learning iterations
iterations = 100000
for _ in tqdm(range(iterations)):
    nn.learn(training_data)

main = RunNN(nn, training_data)
main.plot_gradient_graph()
