import random

import numpy as np

from .activation_function_class import ActivationFunction
from .loss_function_class import LossFunction


class NeuralNetwork:
    def __init__(self, layers_sizes: list[int], activation_functions: list[str],
                 loss_function='squared_error', learning_rate=0.5, epochs=50, mini_batch_size=20,
                 scale_data=True):
        """
        :param layers_sizes: size of the following layers of the network
        :param activation_functions: list of activation functions for each layer
        :param loss_function: name of loss function to calculate error of the output
        :param learning_rate: step for Stochastic Gradient Descend algorithm
        :param epochs: how many iterations through all dataset should be done at the trening
        :param mini_batch_number: number of mini batches for split
        Attributes:
        layers_number - number of layers
        biases - list of vectors being biases for each layer
        weights - list of matrixes with weights for each layer
        a_calculator - object to calculate deravites and function values for activate functions
        l_calculate - object to calculate deravites and function values for loss functions

        The structure of the network is:
        - input is a MNIST number recognition dataset
        - output is numpy array of predicted numbers for the dataset

        Run model.fit to train the network and model.predict to get preditions for your dataset
        """
        self.sizes = layers_sizes
        self.num_layers = len(layers_sizes)
        self.biases = [np.random.uniform(low=(-1/n**0.5), high=(1/n**0.5), size=(n,))
                       # if indx != 0 else np.zeros(shape=(n,))
                       for indx, n in enumerate(layers_sizes[1:])]
        self.weights = [np.random.uniform(low=(-1/x**0.5), high=(1/x**0.5), size=(x, y))
                        # if indx != 0 else np.zeros((x, y))
                        for indx, (x, y) in enumerate(zip(layers_sizes[1:],
                                                      layers_sizes[:-1]))]
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        self.a_calculator = ActivationFunction()
        self.l_calculator = LossFunction()
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.scale_data_flag = scale_data

    def scale_data(self, X: np.ndarray[np.ndarray[int]]) -> np.ndarray[np.ndarray[int]]:
        """scale input data"""
        scaled_data = np.zeros(X.shape)
        minimum = np.min(X)
        maximum = np.max(X)
        for indx, x in enumerate(X):
            scaled_data[indx] = (x - minimum) / (maximum - minimum)
        return scaled_data

    def feedforward(self, a) -> np.ndarray:
        """ Returns the output of the network """
        for w, b, fun in zip(self.weights, self.biases, self.activation_functions):
            z = np.dot(w, a) + b
            a = self.a_calculator.calculate(z, fun)
        return a

    def get_activations_and_z_vectors(self, a) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        get list of all activations and z-vectors for given sample
        """
        activations = [a]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w, fun in zip(self.biases, self.weights, self.activation_functions):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.a_calculator.calculate(z, fun)
            activations.append(a)
        return activations, zs

    def predict(self, X: np.ndarray) -> list[int]:
        """
        get predictions for MNIST dataset
        returns numpy array of predictions for each sample in dataset
        """
        if self.scale_data_flag:
            X = self.scale_data(X)
        preds = [np.argmax(self.feedforward(x)) for x in X]
        return preds

    def create_flatten_mini_batches(self, data: list[tuple]) -> list[list[tuple[np.ndarray[np.ndarray], np.ndarray[int]]]]:
        """
        split dataset x,y into given number of smaller datasets
        smaller datasets are list of pairs (sample_x, sample_y)
        """
        n = len(data)
        random.shuffle(data)
        mini_batches = [data[k:k + self.mini_batch_size] for k in range(0, n, self.mini_batch_size)]
        return mini_batches

    def get_desired_output(self, output_from_database) -> np.ndarray:
        """
        get ideal output from the network
        in 10-elements vector, element with index equal to number to predict should be 1, rest 0
        """
        return np.array([1.0 if indx == output_from_database else 0.0
                         for indx in range(self.sizes[-1])])

    def fit(self, X: np.ndarray[np.ndarray], y) -> None:
        """
        method for training the network. input should be a training dataset
        """
        if self.scale_data_flag:
            X = self.scale_data(X)
        data = [*zip(X, y)]
        for epoch in range(self.epochs):
            mini_batches = self.create_flatten_mini_batches(data)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            print(f'Epoch {epoch + 1} completed.')

    def update_mini_batch(self, mini_batch: list[tuple]) -> None:
        """
        perform SGD on one minibatch to update network's weights and biases
        """
        w_gradient = [np.zeros(w.shape) for w in self.weights]
        b_gradient = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            # calculate dC/dw and dC/db via backpropagation
            w_derivative, b_derivative = self.backpropagation(x, y)
            # sum previous gradients and this one
            w_gradient = [previous + update for previous, update in zip(w_gradient, w_derivative)]
            b_gradient = [previous + update for previous, update in zip(b_gradient, b_derivative)]

        # update current vectors
        # weights = weights - learning_rate * (mean of dC/dw in mini-batch)
        self.weights = [w - (self.learning_rate / len(mini_batch)) * w_grad
                        for w, w_grad in zip(self.weights, w_gradient)]
        # biases = biases - learning_rate * (mean of dC/db in mini-batch)
        self.biases = [b - (self.learning_rate / len(mini_batch)) * b_grad
                       for b, b_grad in zip(self.biases, b_gradient)]

        # o = []
        # do = []
        # for x, y in mini_batch:
        #     o.append(self.feedforward(x))
        #     do.append(self.get_desired_output(y))
        # print(mse(np.array(do), np.array(o)))

    def backpropagation(self, x: np.ndarray, y: float) -> tuple[list, list]:
        """returns dC/dw and dC/db for each layer in the network"""
        # inicialize derivatives to add elements from the back
        w_derivative = [np.zeros(w.shape) for w in self.weights]
        b_derivative = [np.zeros(b.shape) for b in self.biases]
        # get z and activations
        activations, zs = self.get_activations_and_z_vectors(x)

        # get error of the last layer
        output = activations[-1]
        desired_output = self.get_desired_output(y)
        error = self.l_calculator.calculate(output, desired_output, self.loss_function, True) *\
            self.a_calculator.calculate(zs[-1], self.activation_functions[-1], True)
        # error = (output - desired_output) * \
        #     self.a_calculator.calculate(zs[-1], self.activation_functions[-1], True)
        index = -1  # index for hidden layers calculations

        # reshape error and acivation to enable vector multiplication in w_derivative
        reshaped_error = error.reshape((-1, 1))
        reshaped_activation = activations[index-1].reshape((-1, 1))
        w_derivative[index] = np.dot(reshaped_error, reshaped_activation.transpose())
        b_derivative[index] = error

        # derivatives for hidden layers, calculating from the back via backpropagation
        for w, b, func, z in zip(self.weights[::-1], self.biases[::-1], self.activation_functions[-2::-1], zs[-2::-1]):
            index -= 1  # negative index to add elements to w_derivative, b_derivative from back
            func_derivative = self.a_calculator.calculate(z, func, True)
            new_error = np.dot(w.transpose(), error) * func_derivative  # given formula
            error = new_error

            # reshape error and acivation to enable vector multiplication in w_derivative
            reshaped_error = error.reshape((-1, 1))
            reshaped_activation = activations[index-1].reshape((-1, 1))
            w_derivative[index] = np.dot(reshaped_error, reshaped_activation.transpose())
            b_derivative[index] = error

        return w_derivative, b_derivative


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
