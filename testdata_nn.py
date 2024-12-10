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
        self.weights = self.xavier_init((nodes_in, nodes_out))
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
        # Relu function
        # return max(0, weighted_input)
        # Sigmoid function
        return 1 / (1 + np.exp(-weighted_input))

    def node_cost(self, output_activation, expected_activation):
        error = output_activation - expected_activation
        return error * error

    def xavier_init(self, shape):
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape)


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


class NNTrainer:
    def __init__(self, nn, training_data):
        self.nn = nn
        self.training_data = training_data

    def train_nn_from_function(self, to_completion=True, random_point_num=500, iterations=1000, update_iters=500):
        sample_inputs = np.random.rand(random_point_num, 2)

        sample_expected_result = [self.gradient_equation(x) for x in sample_inputs]

        sample_training_data = [DataPoint(sample_inputs[i], [sample_expected_result[i], abs(1 - sample_expected_result[i])]) for i in range(len(sample_inputs))]

        self.training_data = sample_training_data

        if to_completion:

            done_training = False

            iters = 0

            costs = []

            while not done_training:
                training_costs = self.train_model(update_iters, cost_return=True)
                iters += update_iters
                costs.extend(training_costs)
                rounded_sample_result = [1 if self.nn.calculate_outputs(x)[0] > .5 else 0 for x in sample_inputs]
                done_training = self.test_model(sample_inputs, sample_expected_result, rounded_sample_result, costs)

            print(iters)
        else:
            self.train_model(iterations)


    def test_model(self, sample_inputs, expected_result, result, costs):

        # plots a cost line chart, showing how cost decreases over iterations
        self.plot_line([np.arange(len(costs)), costs], title="NN Cost Line Plot", x="Iterations", y="Cost")

        # array of the actual nn predictions
        colormap_colors = [self.nn.calculate_outputs(x)[0] for x in sample_inputs]
        # array of colors highlighting if each point is correctly marked
        rg_colors = ["green" if expected_result[i] == result[i] else "red" for i in range(len(expected_result))]
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
        # sets size of figure
        plt.figure(figsize=figure_size)
        
        # plots line with two input arrays, a color, a linestyle, and a line width
        plt.plot(inputs[0], inputs[1], color=color, linestyle=linestyle, linewidth=linewidth)
        
        # sets title and axis titles
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        
        # adds grid and shows graph
        plt.grid(True)
        plt.show()

    def train_model(self, iterations, cost_return=False):
        # Adjust learning iterations
        costs = []
        for _ in tqdm(range(iterations)):
            self.nn.learn(self.training_data)
            if cost_return:
                costs.append(self.nn.cost(self.training_data))
        return costs



    def gradient_equation(self, coords):
        x, y = coords
        return 1 if x + (y - .2) * (y - .2) > .5 else 0


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
learn_rate = 0.04
nn = NeuralNetwork([2, 3, 2], 64, learn_rate=learn_rate, h=h)  # 2 inputs, 3 hidden nodes, 2 outputs

trainer = NNTrainer(nn, training_data)
trainer.train_nn_from_function(to_completion=True, random_point_num=500, update_iters=500)
