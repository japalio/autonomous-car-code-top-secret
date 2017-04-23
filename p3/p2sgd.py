import numpy as np

def calculateATrue():
	train_n = 100
	test_n = 1000
	d = 100
	sumTrainError = 0
	sumTestError = 0
	for i in range(0, 10):
		X_train = np.random.normal(0,1, size=(train_n,d))
		a = np.random.normal(0,1, size=(d,1))
		y_train = X_train.dot(a) + np.random.normal(0,0.5,size=(train_n,1))
		X_test = np.random.normal(0,1, size=(test_n,d))
		y_test = X_test.dot(a) + np.random.normal(0,0.5,size=(test_n,1))
		

		trainNumerator =  np.linalg.norm(np.dot(X_train, a) - y_train)
		trainDenominator = np.linalg.norm(y_train)

		testNumerator = np.linalg.norm(np.dot(X_test, a) - y_test)
		testDenominator = np.linalg.norm(y_test)

		trainError = trainNumerator / float(trainDenominator)
		testError = testNumerator / float(testDenominator)
		sumTrainError += trainError
		sumTestError += testError


	averagedTrainError = sumTrainError / float(10.0)
	averagedTestError = sumTestError / float(10.0)

	print 'test error with a true: ', averagedTestError
	print 'train error with a true: ', averagedTrainError

def calculateStochasticGradientDescent(step_size):
	train_n = 100
	test_n = 1000
	d = 100
	X_train = np.random.normal(0,1, size=(train_n,d))
	a_true = np.random.normal(0,1, size=(d,1))
	y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
	X_test = np.random.normal(0,1, size=(test_n,d))
	y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))

	a = np.zeros((d, 1))
	
	for iteration in range(0, 1000000):
		index = iteration % 100
		#1 x 100
		aX_train = np.dot(X_train[index], a) - y_train[index]

		#1xd 
	
		gradient = 2*np.dot(X_train[index].T.reshape(d,1), aX_train.reshape(1,1))
		a = a - step_size*gradient
	
	trainNumerator =  np.linalg.norm(np.dot(X_train, a) - y_train)
	trainDenominator = np.linalg.norm(y_train)

	testNumerator = np.linalg.norm(np.dot(X_test, a) - y_test)
	testDenominator = np.linalg.norm(y_test)

	trainError = trainNumerator / float(trainDenominator)
	testError = testNumerator / float(testDenominator)

	return trainError, testError


def calculateAverageError(step_size):
	sumTrainError = 0
	sumTestError = 0
	for i in range(0,10):
		trainError, testError = calculateStochasticGradientDescent(step_size)
		sumTrainError += trainError
		sumTestError += testError

	averagedTrainError = sumTrainError / float(10.0)
	averagedTestError = sumTestError / float(10.0)

	print 'averaged Train Error: ', averagedTrainError
	print 'averaged test error: ', averagedTestError



# stepSizes = [0.00005, 0.0005, 0.005]
# for stepSize in stepSizes:
# 	calculateAverageError(stepSize)

calculateATrue()
	