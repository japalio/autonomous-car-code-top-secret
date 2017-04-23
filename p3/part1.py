import numpy as np
import matplotlib.pyplot as plt
import numpy as numpy
import math 

d = 100 # dimensions of data
n = 1000 # number of data points
X = np.random.normal(0,1, size=(n,d))
a_true = np.random.normal(0,1, size=(d,1))
y = X.dot(a_true) + np.random.normal(0,0.5,size=(n,1))

def calculateClosedForm():
	first_term = np.linalg.inv(np.dot(X.T, X))
	second_term = np.dot(X.T, y)
	a = np.dot(first_term, second_term)

	print calculateSquaredError(a, X, y)


def calculateSquaredError(a, X, y):
	#X is a nxd
	#a is a d x 1
	#y is n x 1 [labels]

	aX = X.dot(a) - y
	error = np.sum(np.square(aX))


	return error



def calculateGradientDescent(step_size):
	a = np.zeros((d, 1))

	for iteration in range(0,20):
		#gradient of the sum = sum of the gradients

		#sum of the gradients:
		#n x 1
		aX = np.dot(X, a) - y
		# aX = X.dot(a) - y

		#n x d
		individualGradients = 2*np.dot(X.T, aX)
		#d x 1  
		# gradient = (np.sum(individualGradients, axis = 0)).reshape((d, 1))
		# print gradient

		a = a - step_size*(individualGradients)
		# print a

	# print calculateSquaredError(a, X, y)
	return calculateSquaredError(a, X, y)

def calculateStochasticGradientDescent(step_size):
	objectiveFunctionValues = []
	a = np.zeros((d, 1))

	for iteration in range(0, 1000):
		
		#1 x 1
		aX = np.dot(X[iteration], a) - y[iteration]

		#1xd 
	
		gradient = 2*np.dot(X[iteration].T.reshape(d,1), aX.reshape(1,1))
		a = a - step_size*gradient
		objectiveFunctionValues.append(math.log(calculateSquaredError(a, X, y)))

	return objectiveFunctionValues

	# print calculateSquaredError(a, X, y)
	# return calculateSquaredError(a, X, y)


calculateClosedForm()
objectiveFuncValsOne = calculateStochasticGradientDescent(0.0005)
objectiveFuncValsTwo = calculateStochasticGradientDescent(0.005)
objectiveFuncValsThree = calculateStochasticGradientDescent(0.01)
plt.xlabel('Iteration Number')
plt.ylabel('Objective Function Value')
iterationList = [x for x in range(1, 1001)]
plt.plot(iterationList, objectiveFuncValsOne, label='step size: 0.0005')
plt.plot(iterationList, objectiveFuncValsTwo, label = 'step size: 0.005')
plt.plot(iterationList, objectiveFuncValsThree, label = 'step size: 0.01')
plt.legend(loc = 'upper left')
plt.show()


# points = []
# points.append(calculateGradientDescent(0.00005))
# points.append(calculateGradientDescent(0.0005))
# points.append(calculateGradientDescent(0.0007))

# print(points)
# plt.plot([0.00005, 0.0005, 0.0007], points)
# # plt.xlabel('Average Classification Error')
# # # plt.ylabel('Average Number of Articles Inspected')
# # plt.title('Classification Error vs Sq Tradeoff')
# plt.show()

