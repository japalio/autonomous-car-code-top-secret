import math
import numpy as np 
from scipy.sparse import csgraph
from numpy import linalg as LA
from pprint import pprint
import matplotlib.pyplot as plt



def plotEigenVector(eigenvector):
	xList = [x for x in range(0, 1495)]
	plt.scatter(xList, eigenvector)
	plt.legend(loc = 'lower left')
	plt.show()



def calculateConductance(indexSet, adjancencyMatrix):
	allOtherIndices = [x for x in range(0,1495)]
	allOtherIndices = set(allOtherIndices)
	allOtherIndices = allOtherIndices - set(indexSet)


	numerator = 0.0
	for i in range(0, len(indexSet)):
		for j in range(0, len(allOtherIndices)):
			numerator += adjancencyMatrix[i][j]

	A_S = 0.0
	for index in range(0, len(indexSet)):
		A_S += np.sum(adjancencyMatrix[index])

	A_V_S = 0.0
	for index in range(0, len(allOtherIndices)):
		A_V_S += np.sum(adjancencyMatrix[index])

	print "Conductance:", numerator/(min(A_S, A_V_S))


def randomSetConducatance(adjancencyMatrix):
	
	calculateConductance(adjancencyMatrix)


def createAdjacancyMatrix():
	adjancencyMatrix = np.zeros((1495,1495))
	with open('cs168mp6.csv') as f:
		for line in f:
			a, b = line.split(",")

			adjancencyMatrix[int(a) - 1][int(b) - 1] += 1


	diagonalMatrix = np.zeros((1495, 1495))
	for x in range(1495):
		diagonalMatrix[x][x] = np.sum(adjancencyMatrix[x])

	laplacian = diagonalMatrix - adjancencyMatrix

	# print diagonalMatrix
	# for x in range(1495):
	# 	print np.sum(laplacian[x])


	#compute the eigenvectors and eigenvalues of the Laplacian
	w, v = np.linalg.eig(laplacian)

	seventhSmallestIndex = np.argsort(w)[6]
	eightSmallestIndex = np.argsort(w)[7]

	seventhEigenVector = v[:,seventhSmallestIndex]

	
	# plotEigenVector(v[:,seventhSmallestIndex])

	set1 = []
	for x in range(0,1495):
		if(seventhEigenVector[x] <= 0 and seventhEigenVector[x] >= -0.012):
			set1.append(x)

	#-0.01317
	print "Number of components in Set S:", len(set1)

	calculateConductance(set1, adjancencyMatrix)
	# plotEigenVector(v[:,eightSmallestIndex])
	

	#number of connected components
	connectedComponent = 0
	for num in w:
		if(num < .00000000001):
			connectedComponent += 1

	print "NUM CONNECTED COMPONENTS", connectedComponent

	# sortedVals = np.sort(w)
	index = 0
	# for item in w:
	# 	# print "ITEM", item
	# 	print type(item)
	# 	if(item == 0.0143040166194):
	# 		print "FIRST INDEX", index
	# 	elif(item == 0.053795652737):
	# 		print "SECOND INDEX", index
	# 	index += 1
		# print item
	
	#0.0143040166194
	#0.053795652737


	# print w.index(0.0143040166194) index = 197
	# print w.index(0.053795652737) index = 196

	
	# print laplacian
	# print diagonalMatrix

createAdjacancyMatrix()
