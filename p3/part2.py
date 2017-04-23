import numpy as np
import matplotlib.pyplot as plt

train_n = 100
test_n = 1000
d = 100

	
def calculateLinearRegressionBaseline():
	sumTrainError = 0
	sumTestError = 0
	for i in range(0, 10):
		X_train = np.random.normal(0,1, size=(train_n,d))
		a_true = np.random.normal(0,1, size=(d,1))
		y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
		X_test = np.random.normal(0,1, size=(test_n,d))
		y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))
		a = np.dot(np.linalg.inv(X_train), y_train)

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

	print 'test error: ', averagedTestError
	print 'train error: ', averagedTrainError

def calculateL2LinearRegression(lambdaVal):
	sumTrainError = 0
	sumTestError = 0
	for i in range(0, 10):
		X_train = np.random.normal(0,1, size=(train_n,d))
		a_true = np.random.normal(0,1, size=(d,1))
		y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
		X_test = np.random.normal(0,1, size=(test_n,d))
		y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))

		inverseThis = np.dot(X_train.T, X_train) + lambdaVal*np.identity(d)
		endPart = np.dot(X_train.T, y_train)
		a = np.dot(np.linalg.inv(inverseThis), endPart)

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
	return averagedTrainError, averagedTestError


calculateLinearRegressionBaseline()
lambdaList = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
trainErrors = []
testErrors = []
for item in lambdaList:
	x, y = calculateL2LinearRegression(item)
	trainErrors.append(x)
	testErrors.append(y)

plt.plot(lambdaList, trainErrors, label = 'normalized train error')
plt.plot(lambdaList, testErrors, label = 'normalized test error')
print 'trainErrors: ', trainErrors
print 'testErrors: ', testErrors
plt.xlabel('Lambda value')
plt.ylabel('Normalized Error')
plt.legend(loc = 'upper left')
plt.show()


trainErrors:  [0.0007208049284952462, 0.0022905641786984112, 0.0082758592065874746, 0.017225704922872835, 0.069571508789751185, 0.29002197176488853, 0.74203317299399585]
testErrors:  [0.45362905077646409, 0.41712233928081616, 0.30151890802359116, 0.26160829801278634, 0.34071961134147921, 0.57857368554560917, 0.86164093530075636]


