import numpy as np
import math

train_n = 100
test_n = 10000
d = 200

def calculateNormalizedError(a, X, y):
	numerator = np.linalg.norm(np.dot(X, a) - y)
	denominator = np.linalg.norm(y)

	return (numerator / float(denominator))



def calculateGradientDescent(step_size, lambdaVal, r):
	X_train = np.random.normal(0,1, size=(train_n,d))
	a_true = np.random.normal(0,1, size=(d,1))
	y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
	X_test = np.random.normal(0,1, size=(test_n,d))
	y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))
	
	#zero initialization
	# a = np.zeros((d, 1))
	a = np.full((d, 1), r)

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

def calculateStochasticGradientDescent(step_size, lambdaVal):
	train_n = 100
	test_n = 10000
	d = 200
	X_train = np.random.normal(0,1, size=(train_n,d))
	a_true = np.random.normal(0,1, size=(d,1))
	y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
	X_test = np.random.normal(0,1, size=(test_n,d))
	y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))
	
	#zero initialization
	#a = np.zeros((d, 1))
	r = .00000000005
	randomNum = np.random.normal(0,1)
	scalingFactor = r / float(np.linalg.norm(randomNum))

	# print np.linalg.norm(randomNum)
	# print 'scaling Factor: ', scalingFactor
	startingPoint = scalingFactor * randomNum
	a = np.full((d, 1), startingPoint)


	for iteration in range(0,50):
		index = iteration % 100
		#gradient of the sum = sum of the gradients

		#sum of the gradients:
		#n x 1
		aX = np.dot(X_train[index], a) - (y_train[index])
		# aX = X.dot(a) - y

		#regularization factor
		regularizationFactor = lambdaVal * (np.linalg.norm(a))
		
		#n x d
		individualGradients = 2*np.dot(X_train[index].T.reshape(d,1), aX.reshape(1,1)) + 2*regularizationFactor
		#d x 1  
		# gradient = (np.sum(individualGradients, axis = 0)).reshape((d, 1))
		# print gradient

		a = a - step_size*(individualGradients)
		# print a

	# print calculateSquaredError(a, X, y)
	return calculateNormalizedError(a, X_test, y_test)


#used gradient descent instead of SGD
#used a regularization factor
#used non-zero intialization

#NEED TO BE CALCULTING **NORMALIZED** TEST ERROR, averaged over 1000 trials
def calculateTestErrorStochastic(step_size, lambdaVal):
	sumTestError = 0.0
	for x in range(1000):
		sumTestError += calculateStochasticGradientDescent(step_size, lambdaVal)
	averageError = sumTestError/ 1000.0
	print averageError

def calculateTestErrorGradient(step_size, lambdaVal, r):
	sumTestError = 0.0
	for x in range(1000):
		sumTestError += calculateGradientDescent(step_size, lambdaVal, r)
	averageError = sumTestError/ 1000.0
	print averageError


rList = [0, 0.1, 0.5, 1, 10, 20, 30]

random_as = []
for r in rList:
    a = np.random.uniform(size=(d, 1))
    an = a / np.sqrt(np.sum(a ** 2, 0))
    random_as.append(r*an)

lambdaVal = 0.060
count = 0
for a in random_as:
	count +=1
	for i in range(30):
		# lambdaVal += i*0.01 
		calculateTestErrorGradient(0.0005, lambdaVal + i*0.01, random_as[0])
		
		print 'finished gradient descent with lambda = ', lambdaVal + i*0.01 ,'and step size 0.0005 and r:', rList[count] 

# calculateTestErrorStochastic(.005, .065)
# print 'sgd with error 0.005 and lambda 0.065 and 100 iterations'

lambdaVal = 0.060
for i in range(30):
	# lambdaVal += i*0.01 
	calculateTestErrorGradient(0.0005, lambdaVal + i*0.01, 0)
	print 'finished gradient descent with lambda = ', lambdaVal + i*0.01 ,'and step size 0.0005'

# calculateTestErrorGradient(0.00045, .065, 0)
# print 'finished gradient descent with lambda = 0.065 and step size 0.00045'