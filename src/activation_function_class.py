import numpy as np


class ActivationFunction:
    def __init__(self) -> None:
        self.functions_dict = {
            'relu': self._relu,
            'relu_derivative': self._relu_derivative,
            'sigmoid': self._sigmoid,
            'sigmoid_derivative': self._sigmoid_derivative,
            'tanh': self._tanh,
            'tanh_derivative': self._tanh_derivative,
            'softplus': self._softplus,
            'softplus_derivative': self._softplus_derivative
        }

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _softplus(self, x):
        return np.log(1 + np.exp(x))

    def _softplus_derivative(self, x):
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        return 1 - np.power(np.tanh(x), 2)

    def calculate(self, x, function_name: str, derivitave=False):
        if not derivitave:
            return self.functions_dict[function_name](x)
        else:
            return self.functions_dict[function_name+'_derivative'](x)
