import numpy as np

d = 100 # dimensions of data
n = 1000 # number of data points
X = np.random.normal(0,1, size=(n,d))
a_true = np.random.normal(0,1, size=(d,1))
y = X.dot(a_true) + np.random.normal(0,0.5,size=(n,1))

def calculateClosedForm():
	first_term = np.linalg.inv(np.matmul(X.T, X))
	second_term = np.matmul(X.T, y)
	a = np.matmul(first_term, second_term)

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


		individualGradients = 2*np.matmul(X.T, aX)
	

		#d x 1  
		# gradient = (np.sum(individualGradients, axis = 0)).reshape((d, 1))
		# print gradient

		a = a - step_size*individualGradients
		# print a

	print calculateSquaredError(a, X, y)



calculateClosedForm()
calculateGradientDescent(0.0006)