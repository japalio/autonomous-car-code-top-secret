import numpy as np
import math


def calculateNormalizedError(a, X, y):
	numerator = np.linalg.norm(np.dot(X, a) - y)
	denominator = np.linalg.norm(y)

	return (numerator / float(denominator))



def calculateGradientDescent(step_size, lambdaVal):
	train_n = 100
	test_n = 10000
	d = 200
	X_train = np.random.normal(0,1, size=(train_n,d))
	a_true = np.random.normal(0,1, size=(d,1))
	y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
	X_test = np.random.normal(0,1, size=(test_n,d))
	y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))
	
	a = np.zeros((d, 1))

	for iteration in range(0,50):
		#gradient of the sum = sum of the gradients

		#sum of the gradients:
		#n x 1
		aX = np.dot(X_train, a) - (y_train)
		# aX = X.dot(a) - y

		#regularization factor
		regularizationFactor = lambdaVal * (np.linalg.norm(a))
		#n x d
		individualGradients = 2*np.dot(X_train.T, aX) + 2*regularizationFactor
		#d x 1  
		# gradient = (np.sum(individualGradients, axis = 0)).reshape((d, 1))
		# print gradient

		a = a - step_size*(individualGradients)
		# print a

	# print calculateSquaredError(a, X, y)
	return calculateNormalizedError(a, X_test, y_test)




#NEED TO BE CALCULTING **NORMALIZED** TEST ERROR, averaged over 1000 trials
def calculateTestError(step_size, lambdaVal):
	sumTestError = 0.0
	for x in range(1000):
		sumTestError += calculateGradientDescent(step_size, lambdaVal)
	averageError = sumTestError/ 1000.0
	print averageError

calculateTestError(.0005, .05)