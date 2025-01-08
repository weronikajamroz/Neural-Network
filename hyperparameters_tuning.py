import numpy as np
import pandas as pd

from src.neural_network import NeuralNetwork
from src.split_dataset import split_dataset_train_test_validate
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


# prepare datasets
digits = load_digits(as_frame=True)
digits_X = digits['data']
digits_y = digits['target']
(X_train, y_train, X_validate, y_validate,
 X_test, y_test) = split_dataset_train_test_validate(digits_X, digits_y)


def tune_learning_rate():
    learning_rate = [0., 0.1, 0.2, 0.4, 0.5, 0.55, 0.6, 0.8, 1., 1.5, 2., 3., 4.]
    acc = []
    model = NeuralNetwork([64, 16, 16, 10], ['sigmoid', 'sigmoid', 'sigmoid'],
                          'squared_error', 0.55, 50, 15, False)
    for lr in learning_rate:
        model.learning_rate = lr
        model.fit(X_train, y_train)
        preds = model.predict(X_validate)
        acc.append(accuracy_score(y_validate, preds))
        print(f'--------- Lr = {lr} completed ---------')

    plt.plot(learning_rate, acc, color='navy', zorder=2)
    plt.title('Correlation between learning rate and predictions accuracy')
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey', zorder=1)
    plt.show()


def tune_epochs():
    epochs = [0, 25, 50, 75, 100, 15, 200, 300, 400, 500, 750, 1000]
    acc_train = []
    acc_val = []
    model = NeuralNetwork([64, 16, 16, 10], ['sigmoid', 'sigmoid', 'sigmoid'],
                          'squared_error', 0.55, 100, 15, False)
    for epoch in epochs:
        model.epochs = epoch
        model.fit(X_train, y_train)
        preds_train = model.predict(X_train)
        a_t = accuracy_score(y_train, preds_train)
        acc_train.append(a_t)
        preds_val = model.predict(X_validate)
        a_v = accuracy_score(y_validate, preds_val)
        acc_val.append(a_v)
        print(f' --------- Epochs {epoch} completed --------- ')

    plt.plot(epochs, acc_train, color='lightsteelblue', zorder=2, label='Train set')
    plt.plot(epochs, acc_val, color='navy', zorder=2, label='Validation set')
    plt.title('Correlation between training epochs and predictions accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey', zorder=1)
    plt.show()


def tune_mini_batch_size():
    mini_batch_size = [1, 2, 4, 6, 8, 10, 20, 40, 70, 100, 200, 300]
    acc = []
    model = NeuralNetwork([64, 16, 16, 10], ['sigmoid', 'sigmoid', 'sigmoid'],
                          'squared_error', 0.55, 50, 10, False)
    for mbs in mini_batch_size:
        model.mini_batch_size = mbs
        model.fit(X_train, y_train)
        preds = model.predict(X_validate)
        acc.append(accuracy_score(y_validate, preds))
        print(f'--------- MBS = {mbs} completed ---------')

    plt.plot(mini_batch_size, acc, color='navy', zorder=2)
    plt.title('Correlation between mini batch size and predictions accuracy')
    plt.xlabel('Mini batch size')
    plt.ylabel('Accuracy')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey', zorder=1)
    plt.show()


def tune_layers_lr():
    learning_rates = [0.001, 0.1, 0.5, 2, 6]
    results = []
    X = np.arange(10, 100, 10, dtype=int)
    for lr in learning_rates:
        lr_results = []
        for x in X:
            model = NeuralNetwork([64, x, x, x, 10], ['softplus', 'softplus', 'softplus', 'softplus'], 'squared_error',
                                  learning_rate=lr, epochs=100, mini_batch_size=10)
            model.fit(X_train, y_train)
            preds = model.predict(X_validate)
            lr_results.append(accuracy_score(preds, y_validate))
        results.append(lr_results)

    for indx, lr_result in enumerate(results):
        plt.plot(X, lr_result, label=f'{learning_rates[indx]}')
    plt.title("Three layer for different learning rates")
    plt.xlabel("size of the layer")
    plt.ylabel("accurancy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    tune_mini_batch_size()
