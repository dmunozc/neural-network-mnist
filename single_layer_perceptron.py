import numpy as np
from numpy import random
from numpy import array
import csv
import sys

#This function computes new perceptron weights of a given data set and weights
#and learning rate
def recalculateWeights(trainingData, initialWeights, learningRate):
  newWeights = initialWeights.copy()
  #iterating over every row of the data set
  for row in trainingData:
    #normalize the input data and add the bias value
    xs = array(list(map(int,row[1:])))/255
    xs = np.concatenate(([1], xs))
    #calculate the dot product of the input and weights
    w_x = (newWeights * xs).sum(axis=1)
    #for each perceptron calculate the required learning rate
    #and use it to calculate the new weights
    for i in range(len(newWeights)):
      y = 1 if (w_x[i]) > 0 else 0
      t = 1 if i == int(row[0]) else 0
      lR = learningRate * (t - y)
      newWeights[i] = newWeights[i] +  lR *  xs
  return newWeights

#this function generates the 2 dimensional confusion matrix given a data set 
#and weights
def computeConfusionMatrix(testData, weights):
  #initialize a 10x10 matrix
  matrix = [[0 for _ in range(10)] for _ in range(10)]
  #iterating over every row of the data set
  for row in testData:
    #normalize the input data and add the bias value
    xs = array(list(map(int,row[1:])))/255
    xs = np.concatenate(([1],xs))
    #calculate the dot product of the input and weights
    w_x = (weights * xs).sum(axis=1)
    #The maximum value of this dot product is the number predicted in the
    #perceptrons
    predictionNumber = np.where(w_x == max(w_x))[0][0]
    #append that value to the confusion matrix. which is [expected value][predicted value]
    matrix[int(row[0])][predictionNumber] = matrix[int(row[0])][predictionNumber] + 1
  return matrix
  
#This function computes the accuracy of a given data set and weights
def computeAccuracy(trainingData, weights):
  accuracy = 0
  totalLines = 0
  #iterating over every row of the data set
  for row in trainingData:
    #normalize the input data and add the bias value
    xs = array(list(map(int,row[1:])))/255
    xs = np.concatenate(([1], xs))
    #calculate the dot product of the input and weights
    w_x = (weights * xs).sum(axis=1)
    #The maximum value of this dot product is the number predicted in the
    #perceptrons
    predictionNumber = np.where(w_x == max(w_x))[0][0]
    totalLines =totalLines + 1 
    #If that matches the expected value from the data set it is added
    #to the total accuracy
    if predictionNumber == int(row[0]):
      accuracy = accuracy + 1
  return accuracy/totalLines

def main(trainingFilePath, testFilePath, learningRate=0.1, epochs=50):
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
  #create a random 10x78 =5 weight vector
  weights = [random.uniform(-0.5,0.5,785) for _ in range(10)]
  
  
  #iterate over the number of epochs
  #for each epoch print the accurace of the test and training sets
  #then recalculate the weights using the training formula for perceptrons
  #and save it for using in the next epoch
  for i in range(epochs):
    print("Epoch", i)
    print(computeAccuracy(training, weights))
    print(computeAccuracy(test, weights))
    weights = recalculateWeights(training, weights, learningRate)
  #once the training is done, print the confusion matrix
  print(computeConfusionMatrix(test, weights))
  
  
if __name__ == '__main__':
  if len(sys.argv) < 5:
    print("Usage", sys.argv[0], "mnist_train.csv mnist_test.csv 0.1 50")
  trainingFilePath = sys.argv[1]
  testFilePath = sys.argv[2]
  learningRate = float(sys.argv[3])
  epochs = int(sys.argv[4])
  main(trainingFilePath, testFilePath, learningRate, epochs)
