import numpy as np
import matplotlib.pyplot as plt
import math

train_n = 100
test_n = 1000
d = 100
def calculateStochasticGradientDescent(step_size, a):
	
	X_train = np.random.normal(0,1, size=(train_n,d))
	a_true = np.random.normal(0,1, size=(d,1))
	y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
	X_test = np.random.normal(0,1, size=(test_n,d))
	y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))

	# randomNum = np.random.normal(0,1)
	# scalingFactor = r / float(np.linalg.norm(randomNum))


	# print np.linalg.norm(randomNum)
	# print 'scaling Factor: ', scalingFactor
	# startingPoint = scalingFactor * randomNum
	a = np.full((d, 1), a)
	print 'starting point norm: ', np.linalg.norm(a)

	
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

def calculateAverageError(step_size, r):
	sumTrainError = 0
	sumTestError = 0
	for i in range(0,10):
		trainError, testError = calculateStochasticGradientDescent(step_size, r)
		sumTrainError += trainError
		sumTestError += testError

	averagedTrainError = sumTrainError / float(10.0)
	averagedTestError = sumTestError / float(10.0)

	print 'averaged Train Error: ', averagedTrainError
	print 'averaged test error: ', averagedTestError
	return averagedTrainError, averagedTestError


rList = [0, 0.1, 0.5, 1, 10, 20, 30]


random_as = []
for r in rList:
    a = np.random.uniform(size=(d, 1))
    an = a / np.sqrt(np.sum(a ** 2, 0))
    random_as.append(r*an)

trainErrors = []
testErrors = []
for a in random_as:
	x, y = calculateAverageError(0.00005, a)
	trainErrors.append(x)
	testErrors.append(y)

plt.scatter(rList, trainErrors, label = 'normalized train error', color='blue')
plt.scatter(rList, testErrors, label = 'normalized test error',color='red')
print 'trainErrors: ', trainErrors
print 'testErrors: ', testErrors
plt.xlabel('r value')
plt.ylabel('Normalized Error')
plt.legend(loc = 'upper left')
plt.show()


# trainErrors:  [0.013132304255506006, 0.013736828125278721, 0.014055662460894375, 0.016996822866480271, 0.11474671201807782, 0.19534064863467582, 0.26835483774811081]
# testErrors:  [0.24419939163559984, 0.20807622121370378, 0.2200703851925862, 0.30235362276457056, 2.3508882682948289, 3.9513058894992126, 5.847254355295167]


