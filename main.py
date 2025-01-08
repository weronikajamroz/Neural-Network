from src.neural_network import NeuralNetwork
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from src.split_dataset import split_dataset_train_test_validate


# prepare datasets
digits = load_digits(as_frame=True)
digits_X = digits['data']
digits_y = digits['target']
(X_train, y_train, X_validate, y_validate,
 X_test, y_test) = split_dataset_train_test_validate(digits_X, digits_y)


# train model
model = NeuralNetwork([64, 16, 16, 10], ['relu', 'relu', 'relu'],
                      'squared_error', 0.001, 1000,
                      10, True)
model.fit(X_train, y_train)


# predict
preds = model.predict(X_test)


# visualize scores
acc = accuracy_score(y_test, preds)
print(f'Accuracy on Digits recognition: {acc}')
disp = ConfusionMatrixDisplay.from_predictions(y_test, preds)
disp.figure_.suptitle("Confusion Matrix")

plt.show()
