"""Demonstration of a multi-layer neural network classification.

It implements one hidden layer.

It uses the mnist dataset for training and testing. It achieves a 91%
out of sample accuracy after 1 epoch, and 93% accuracy after 10 epochs.
"""
import numpy as np
from numpy import random
from numpy import array
import pandas as pd
import argparse


def confusion_matrix(data, in_to_hid_weights, hid_to_out_weights):
    """Generate 2 dimensional confusion matrix."""
    # Initialize a 10x10 matrix.
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    for row in data:
        # Normalize the input data and add the bias value.
        xs = row[1:] / 255
        xs = np.concatenate(([1], xs))
        # Calculate the sigmoid function of input to hidden values.
        h_x = 1 / (1 + (np.exp(-np.dot(in_to_hid_weights, xs))))
        h_x = np.concatenate(([1], h_x))
        # Calculate the sigmoid function of hidden to output values.
        o_h = 1 / (1 + (np.exp(-np.dot(hid_to_out_weights, h_x))))
        # The maximum value of the output is the prediction of the neural
        # network.
        prediction = np.where(o_h == max(o_h))[0][0]
        # Append that value to the confusion matrix. It is [expected
        # value][predicted value].
        matrix[int(row[0])][prediction] += 1
    return matrix


def recalculate_weights(
    data,
    in_to_hid_weights,
    hid_to_out_weights,
    learning_rate,
    momentum,
    pos_target=0.9,
    neg_target=0.1,
):
    """Return new input and hidden weights.

    It implements momentum when calculating backpropagation.
    """
    new_in_to_hid_weights = in_to_hid_weights.copy()
    prev_in_to_hid_weights = new_in_to_hid_weights
    new_hid_to_out_weights = hid_to_out_weights.copy()
    prev_hid_to_out_weights = new_hid_to_out_weights
    for row in data:
        # Normalize the input data and add the bias value.
        xs = row[1:] / 255
        xs = np.concatenate(([1], xs))
        # Calculate the sigmoid function of input to hidden values.
        h_x = 1 / (1 + (np.exp(-np.dot(new_in_to_hid_weights, xs))))
        h_x = np.concatenate(([1], h_x))
        # Calculate the sigmoid function of hidden to output values.
        o_h = 1 / (1 + (np.exp(-np.dot(new_hid_to_out_weights, h_x))))
        # Create matrix of target values. Each column has only 1
        # positive value
        ts = np.full((10, ), neg_target)
        ts[int(row[0])] = pos_target
        # Calculate the change of weight needed for output to hidden.
        delta_o = o_h * (1 - o_h) * (ts - o_h)
        delta_h = []
        # Calculate the weight change of needed for hidden to input values.
        #
        # Need to iterate over all previous input to hidden weights to
        # calculate the new weight values.
        for j in range(len(new_in_to_hid_weights)):
            s = np.dot(new_hid_to_out_weights[:, j + 1], delta_o)
            delta_h.append(h_x[j + 1] * (1 - h_x[j + 1]) * s)
        delta_h = array(delta_h)
        # calculate the momentum values for the weights.
        momentum_o = momentum * (new_hid_to_out_weights - prev_hid_to_out_weights)
        momentum_h = momentum * (new_in_to_hid_weights - prev_in_to_hid_weights)
        # keep a copy of the current weights; which in the next iteration
        # are used to calculate the momentum.
        prev_hid_to_out_weights = new_hid_to_out_weights.copy()
        prev_in_to_hid_weights = new_in_to_hid_weights.copy()

        # calculate the new weights.
        new_hid_to_out_weights += (
            delta_o[:, np.newaxis] * learning_rate * h_x
        ) + momentum_o
        new_in_to_hid_weights += (
            delta_h[:, np.newaxis] * learning_rate * xs + momentum_h
        )

    return new_in_to_hid_weights, new_hid_to_out_weights


def compute_accuracy(data, in_to_hid_weights, hid_to_out_weights):
    """Return accuracy of predictions."""
    accuracy = 0
    for row in data:
        # normalize the input data and add the bias value.
        xs = row[1:] / 255
        xs = np.concatenate(([1], xs))
        # calculate the sigmoid function of input to hidden values.
        h_x = 1 / (1 + (np.exp(-np.dot(in_to_hid_weights, xs))))
        h_x = np.concatenate(([1], h_x))
        # calculate the sigmoid function of hidden to output values.
        o_h = 1 / (1 + (np.exp(-np.dot(hid_to_out_weights, h_x))))
        # The maximum value of the output is the prediction of the
        # neural network.
        prediction = np.where(o_h == max(o_h))[0][0]
        # If that matches the expected value from the data set it is added
        # to the total accuracy.
        if prediction == int(row[0]):
            accuracy = accuracy + 1
    return accuracy / len(data)


def main(
    training_file_path,
    test_file_path,
    learning_rate=0.1,
    epochs=50,
    hidden_layers=20,
    momentum=0.5,
):
    """
    Prints accuracy after every epoch, and confusion matrix
    at the end of training.
    """
    # Open CSV training and test files.
    trainingData = pd.read_csv(training_file_path, index_col=None, header=None)
    testData = pd.read_csv(test_file_path, index_col=None, header=None)
    training = trainingData.values
    test = testData.values

    # Set training conditions.
    pos_target = 0.9
    neg_target = 0.1

    # Create a random weight vector that connects to all hidden layers.
    in_to_hid_weights = np.random.uniform(-0.05, 0.05, size=(hidden_layers, trainingData.shape[1]))
    # Create a random weight vector that connects to output weights.
    hid_to_out_weights = np.random.uniform(-0.05, 0.05, size=(10, hidden_layers+1))
    # For each epoch print the accuracy of the test and training sets.
    #
    # Recalculate the weights using the training formula for neural
    # networks and back propagation and save it for using in the next epoch.
    for i in range(epochs):
        print("Epoch", i + 1)
        in_to_hid_weights, hid_to_out_weights = recalculate_weights(
            training,
            in_to_hid_weights,
            hid_to_out_weights,
            learning_rate,
            momentum,
            pos_target,
            neg_target,
        )
        print(
            "Training accuracy",
            compute_accuracy(training, in_to_hid_weights, hid_to_out_weights),
        )
        print(
            "Test accuracy",
            compute_accuracy(test, in_to_hid_weights, hid_to_out_weights),
        )

    # Once the training is done, print the confusion matrix.
    print("Confusion matrix")
    for row in confusion_matrix(test, in_to_hid_weights, hid_to_out_weights):
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
    parser.add_argument("-m", dest="mom", help="momentum", default=0.5)
    parser.add_argument("-hu", dest="hu", help="hidden_units", default=10)
    parser.add_argument("-e", dest="epochs", help="epochs", default=10)
    args = parser.parse_args()
    main(
        args.train,
        args.test,
        learning_rate=float(args.lr),
        epochs=int(args.epochs),
        hidden_layers=int(args.hu),
        momentum=float(args.mom),
    )
