import numpy as np
import matplotlib.pyplot as plt
import math

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

	print 'baseline test error: ', averagedTestError
	print 'baseline train error: ', averagedTrainError

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




#2a)
# calculateLinearRegressionBaseline()

# 2b)
lambdaList = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
loglambdalist = []
for lam in lambdaList:
	lam = math.log(lam)
	loglambdalist.append(lam)

trainErrors = []
testErrors = []
for item in lambdaList:
	x, y = calculateL2LinearRegression(item)
	trainErrors.append(x)
	testErrors.append(y)

plt.scatter(loglambdalist, trainErrors, label = 'normalized train error', color='red')
plt.scatter(loglambdalist, testErrors, label = 'normalized test error', color='blue')
print 'trainErrors: ', trainErrors
print 'testErrors: ', testErrors
plt.xlabel('Lambda value (log)')
plt.ylabel('Normalized Error')
plt.legend(loc = 'upper left')
plt.show()


#results: 
# lambdaList = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]	
# trainErrors:  [0.0007208049284952462, 0.0022905641786984112, 0.0082758592065874746, 0.017225704922872835, 0.069571508789751185, 0.29002197176488853, 0.74203317299399585]
# testErrors:  [0.45362905077646409, 0.41712233928081616, 0.30151890802359116, 0.26160829801278634, 0.34071961134147921, 0.57857368554560917, 0.86164093530075636]


# trainErrors:  [0.0012096184037917236, 0.0035525681699983039, 0.0049830114674279825, 0.014670830128581724, 0.068711002109696023, 0.29975385035652091, 0.74153645986237615]
# testErrors:  [0.5488815438709459, 0.44884408979692553, 0.28979755429576731, 0.2234280611757804, 0.34307317079560018, 0.54973011701524466, 0.86593358288029554]