import numpy as np
from numpy import random
from numpy import array
import csv
import sys

#this function generates the 2 dimensional confusion matrix given a data set 
#and weights
def computeConfusionMatrix(testData, inputToHiddenWeights, hiddenToOutputWeights):
  #initialize a 10x10 matrix
  matrix = [[0 for _ in range(10)] for _ in range(10)]
  #iterating over every row of the data set
  for row in testData:
    #normalize the input data and add the bias value
    xs = array(list(map(int,row[1:])))/255
    xs = np.concatenate(([1], xs))
    #calculate the sigmoid function of input to hidden values
    h_x = array(list(map(lambda z: 1/(1+(np.exp(-z))), 
                         (inputToHiddenWeights * xs).sum(axis=1))))
    h_x = np.concatenate(([1],h_x))
    #calculate the sigmoid function of hidden to output values
    o_h = array(list(map(lambda z: 1/(1+(np.exp(-z))), 
                         (hiddenToOutputWeights * h_x).sum(axis=1))))
    #The maximum value of the output is the prediction of the neural network
    predictionNumber = np.where(o_h == max(o_h))[0][0]
    #append that value to the confusion matrix. which is [expected value][predicted value]
    matrix[int(row[0])][predictionNumber] = matrix[int(row[0])][predictionNumber] + 1
  return matrix

#This function computes new weights for the neural network using backpropagation
#and momentum
def recalculateWeights(trainingData, inputToHiddenWeights, hiddenToOutputWeights,
                       positiveTargetValue, negativeTargetValue, learningRate, momentum):
  newInputToHiddenWeights = inputToHiddenWeights.copy()
  prevInputToHiddenWeights = newInputToHiddenWeights
  newHiddenToOutputWeights = hiddenToOutputWeights.copy()
  prevHiddenToOutputWeights = newHiddenToOutputWeights
  #iterating over every row of the data set
  for row in trainingData:
    #normalize the input data and add the bias value
    xs = array(list(map(int,row[1:])))/255
    xs = np.concatenate(([1], xs))
    #calculate the sigmoid function of input to hidden values
    h_x = array(list(map(lambda z: 1/(1+(np.exp(-z))), 
                         (newInputToHiddenWeights * xs).sum(axis=1))))
    h_x = np.concatenate(([1],h_x))
    #calculate the sigmoid function of hidden to output values
    o_h = array(list(map(lambda z: 1/(1+(np.exp(-z))), 
                         (newHiddenToOutputWeights * h_x).sum(axis=1))))
    #create matrix  of target values, each column has only 1 positive value value
    ts = array([(positiveTargetValue if x == int(row[0]) else negativeTargetValue) for x in range(10)])
    #calculate the change of weight needed for output to hidden
    delta_o = o_h * (1-o_h) * (ts-o_h)
    delta_h = []
    #calculate the change of needed for hidden to input values
    #need to iterate over all previous input to hidden weights to calculate
    #the new weight values
    for j in range(len(newInputToHiddenWeights)):
      s = 0
      for k in range(len(delta_o)):
        s = s + newHiddenToOutputWeights[k][j+1] * delta_o[k]
      delta_h.append(h_x[j+1] * (1 - h_x[j+1]) * s)
    delta_h  = array(delta_h)
    #calculate the momentum values for the weights
    momentum_o = momentum * np.subtract(newHiddenToOutputWeights, prevHiddenToOutputWeights)
    momentum_h = momentum * np.subtract(newInputToHiddenWeights, prevInputToHiddenWeights)
    
    #keep a copy of the current weights, which in the next iteration are used 
    #to calculate the momentum
    prevHiddenToOutputWeights = newHiddenToOutputWeights.copy()
    prevInputToHiddenWeights = newInputToHiddenWeights.copy()
    
    #calculate the new weights
    newHiddenToOutputWeights = newHiddenToOutputWeights + \
              ((delta_o[:, np.newaxis] * learningRate * h_x) + momentum_o)
    newInputToHiddenWeights = newInputToHiddenWeights + \
              (delta_h[:, np.newaxis]  * learningRate * xs  + momentum_h)

  return newInputToHiddenWeights, newHiddenToOutputWeights
  
#This function computes the accuracy of a given data set and weights
#for a neural network with one hidden layer
def computeAccuracy(data, inputToHiddenWeights, hiddenToOutputWeights):
  accuracy = 0
  totalLines = 0
  #iterating over every row of the data set
  for row in data:
    #normalize the input data and add the bias value
    xs = array(list(map(int,row[1:])))/255
    xs = np.concatenate(([1], xs))
    #calculate the sigmoid function of input to hidden values
    h_x = array(list(map(lambda z: 1/(1+(np.exp(-z))), 
                         (inputToHiddenWeights * xs).sum(axis=1))))
    h_x = np.concatenate(([1],h_x))
    #calculate the sigmoid function of hidden to output values
    o_h = array(list(map(lambda z: 1/(1+(np.exp(-z))), 
                         (hiddenToOutputWeights * h_x).sum(axis=1))))
    #The maximum value of the output is the prediction of the neural network
    predictionNumber = np.where(o_h == max(o_h))[0][0]
    totalLines = totalLines + 1 
    #If that matches the expected value from the data set it is added
    #to the total accuracy
    if predictionNumber == int(row[0]):
      accuracy = accuracy + 1
  return accuracy/totalLines

def main(trainingFilePath, testFilePath, learningRate=0.1, epochs=50, 
         hiddenLayers=20, momentum=0.5):
  #Open CSV training and test files
  trainingFile = open(trainingFilePath)
  testFile = open(testFilePath)
  trainingData = csv.reader(trainingFile, delimiter=',')
  testData = csv.reader(testFile, delimiter=',')
  training = []
  test = []
  #Add the CSV data into arrays since the data is going to be used multiple
  #times
  for row in trainingData:
    training.append(row)
  for row in testData:
    test.append(row)
  
  #set training conditions  
  positiveTargetValue = 0.9
  negativeTargetValue = 0.1
  #create a random weight vector that connects to all hidden layers
  inputToHiddenWeights = [random.uniform(-0.05, 0.05, 785) for _ in range(hiddenLayers)]
  #create a random weight vector that connects to output weights
  hiddenToOutputWeights = [random.uniform(-0.05, 0.05, hiddenLayers+1) for _ in range(10)]
  #iterate over the number of epochs
  #for each epoch print the accuracy of the test and training sets
  #then recalculate the weights using the training formula for neural networks 
  #and back propagation
  #and save it for using in the next epoch
  for i in range(epochs):
    print("Epoch",i)
    print(computeAccuracy(training, inputToHiddenWeights,hiddenToOutputWeights))
    print(computeAccuracy(test, inputToHiddenWeights,hiddenToOutputWeights))
    inputToHiddenWeights,hiddenToOutputWeights = \
        recalculateWeights(training, inputToHiddenWeights, hiddenToOutputWeights,
                           positiveTargetValue, negativeTargetValue, learningRate, 
                           momentum)
  #once the training is done, print the confusion matrix
  print(computeConfusionMatrix(test,inputToHiddenWeights,hiddenToOutputWeights))
  
  
if __name__ == '__main__':
  if len(sys.argv) < 5:
    print("Usage", sys.argv[0], "mnist_train.csv mnist_test.csv 0.1 50")
  trainingFilePath = sys.argv[1]
  testFilePath = sys.argv[2]
  learningRate = float(sys.argv[3])
  epochs = int(sys.argv[4])
  main(trainingFilePath, testFilePath, learningRate, epochs)
