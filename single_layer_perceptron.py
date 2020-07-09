"""Demonstration of a single layer perceptron classification.

It uses the mnist dataset for training and testing. It achieves an 83%
out of sample accuracy after 1 epoch, and 86% accuracy after 10 epochs.
"""

import numpy as np
from numpy import random
from numpy import array
import csv
import argparse


def normalize_and_dot_product(row, weights):
    """Normalize input, add bias, and calculate dot product."""
    # Normalize the input data and add the bias value.
    xs = array(list(map(int, row[1:]))) / 255
    xs = np.concatenate(([1], xs))
    # Calculate the dot product of the input and weights.
    return (weights * xs).sum(axis=1)


def recalculate_weights(training, weights, learning_rate):
    """Return new perceptron weights."""
    new_weights = weights.copy()
    for row in training:
        # Normalize the input data and add the bias value.
        xs = array(list(map(int, row[1:]))) / 255
        xs = np.concatenate(([1], xs))
        # Calculate the dot product of the input and weights.
        w_x = (new_weights * xs).sum(axis=1)
        # For each perceptron calculate the required learning rate and
        # use it to calculate the new weights.
        for i in range(len(new_weights)):
            y_pred = 1 if (w_x[i]) > 0 else 0
            y_actual = 1 if i == int(row[0]) else 0
            new_weights[i] += learning_rate * (y_actual - y_pred) * xs
    return new_weights


def confusion_matrix(data, weights):
    """Generate 2 dimensional confusion matrix."""
    # Initialize a 10x10 matrix.
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    for row in data:
        # Normalize the input data and calculate the dot product of the
        # input and weights.
        w_x = normalize_and_dot_product(row, weights)
        # The maximum value of this dot product is the number predicted
        # in the perceptrons.
        prediction = np.where(w_x == max(w_x))[0][0]
        # Append that value to the confusion matrix. It is [expected
        # value][predicted value]
        matrix[int(row[0])][prediction] += 1
    return matrix


def get_accuracy(training, weights):
    """Return accuracy of predictions."""
    accuracy = 0
    for row in training:
        # Normalize the input data and calculate the dot product of the
        # input and weights.
        w_x = normalize_and_dot_product(row, weights)
        # The maximum value of this dot product is the number predicted
        # in the perceptrons.
        prediction = np.where(w_x == max(w_x))[0][0]
        # If that matches the expected value from the data set it is added
        # to the total accuracy.
        if prediction == int(row[0]):
            accuracy += 1
    return accuracy / len(training)


def main(training_file_path, test_file_path, learning_rate=0.1, epochs=50):
    """
    Prints accuracy after every epoch, and confusion matrix
    at the end of training.
    """
    # Open CSV training and test files.
    training_data = csv.reader(open(training_file_path), delimiter=",")
    test_data = csv.reader(open(test_file_path), delimiter=",")
    training = []
    test = []
    # Add the CSV data into arrays since the data is going to be used
    # multiple times.
    for row in training_data:
        training.append(row)
    for row in test_data:
        test.append(row)
    # Create a random weight vector.
    weights = [random.uniform(-0.5, 0.5, 785) for _ in range(10)]
    # For each epoch print the accuracy of the test and training sets.
    #
    # Recalculate the weights using the training formula for
    # perceptrons and save it for using in the next epoch.
    for i in range(epochs):
        print("Epoch", i + 1)
        weights = recalculate_weights(training, weights, learning_rate)
        print("Training Accuracy", get_accuracy(training, weights))
        print("Test Accuracy", get_accuracy(test, weights))
    # Once the training is done, print the confusion matrix.
    print("Confusing Matrix")
    for row in confusion_matrix(test, weights):
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-train",
        dest="train",
        help="mnist train dataset",
        default="mnist_train.csv",
    )
    parser.add_argument(
        "-test",
        dest="test",
        help="mnist test dataset",
        default="mnist_test.csv",
    )
    parser.add_argument("-lr", dest="lr", help="learning rate", default=0.1)
    parser.add_argument("-e", dest="epochs", help="epochs", default=10)
    args = parser.parse_args()
    main(args.train, args.test, float(args.lr), int(args.epochs))
