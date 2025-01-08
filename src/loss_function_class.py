import numpy as np


class LossFunction:
    def __init__(self) -> None:
        self.functions_dict = {
            'entropy': self._entropy,
            'entropy_derivative': self._entropy_derivative,
            'squared_error': self._squared_error,
            'squared_error_derivative': self._squared_derivative
        }

    def _entropy(self, output: np.ndarray[float], desired_output: np.ndarray[float]) -> np.ndarray[float]:
        return desired_output*np.log(output) + (1-desired_output)*np.log(1-output)

    def _squared_error(self, output: np.ndarray[float], desired_output: np.ndarray[float]) -> np.ndarray[float]:
        return 0.5*(output - desired_output)**2

    def _entropy_derivative(self, output: np.ndarray[float], desired_output: np.ndarray[float]) -> np.ndarray[float]:
        return (output - desired_output)/(output*(1-output))

    def _squared_derivative(self, output: np.ndarray[float], desired_output: np.ndarray[float]) -> np.ndarray[float]:
        return output - desired_output

    def calculate(self, output: np.ndarray, desired_output: np.ndarray,
                  function_name, derivative=False):
        if not derivative:
            return self.functions_dict[function_name](output, desired_output)
        else:
            return self.functions_dict[function_name+'_derivative'](output, desired_output)
