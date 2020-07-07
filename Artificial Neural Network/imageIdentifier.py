import numpy as np
import numpy.random as rn
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import Artificial_neuralNetwork_module as neuralNetwork ## You have to import the module ANN that I've made 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as img


def setup():
	print('[loading pictures to be feed]')
	digits = load_digits()

	# xscale is the normalizer
	print('\n[normalizing data]')
	x_scale = StandardScaler()

	x = x_scale.fit_transform(digits.data)

	# the desired output
	y = digits.target

	## splitting training sets into data sets
	print('\n[splitting data]')
	xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.4)

	# Multilayer perceptron Neural Network structure
	nn = neuralNetwork.neuralNetwork(64,30,10)

	return xtrain, xtest, ytrain, ytest, nn

def training(numberOfTraining, nn):
    print('\n[Training]')
    for i in range(1, numberOfTraining):
    	for j in range(len(xtrain)):
    		train = nn.trainAlgo(xtrain[j],ytrain[j])


def feedForward(nn, xtest, ytest):
	print('\n[Test]')
	nn.setWantPredict(True)
	train = nn.feedForward(xtest[0])
	print('The target: ',ytest[0],' | guess: ',train, end=' ')
	if train != ytest[0]:
	  print('ERROR')
	image = xtest[0].reshape(8, 8)

	return image

def computeAccuracy(nn, trainNumber, xtest, ytest):
	targetPred = nn.predictIt(xtest)
	return f'[Computing the accuracy]\nNumber of training : {trainNumber} \nAccuracy : {nn.accuracyIs(ytest,targetPred)}'


if __name__ == '__main__':
	number_of_training = 100

	xtrain, xtest, ytrain, ytest, nn = setup()
	
	training(number_of_training, nn)
	
	image = feedForward(nn, xtest, ytest)
	
	print(computeAccuracy(nn, number_of_training, xtest, ytest))
	
	plt.imshow(image, cmap='gray')
	
	plt.show()