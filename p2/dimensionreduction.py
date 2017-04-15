import csv 
import numpy as np
from scipy.sparse import csr_matrix 
import math
import matplotlib.pyplot as plt
import warnings 

def read_data():
	reader = csv.reader(open('data/data50.csv', 'rb'))
	#matrix create here
	global matrix
	matrix = np.zeros((1000, 61067))
	for row in reader:
		articleid = int(row[0]) - 1
		wordid = int(row[1]) - 1
		wordcount = int(row[2])
	
		matrix[articleid][wordid] = wordcount 
	matrix = csr_matrix(matrix)
	return matrix 

def calculateCosineSimilarity(aWordVector, bWordVector):

	aNorm = np.linalg.norm(aWordVector)
	bNorm = np.linalg.norm(bWordVector)
	dotProduct = np.dot(aWordVector, bWordVector.T)
	similarity = dotProduct / float((aNorm * bNorm))
	return similarity

def greatestSimiliarity(hostArticle, matrix):
	currMax = 0.0
	maxArticle = None

	for article in range(matrix.shape[0]):
		if(hostArticle != article):
			similarity = calculateCosineSimilarity(matrix[hostArticle], matrix[article])
			if(similarity > currMax):
				currMax = similarity
				maxArticle = article
	return maxArticle

	
def baselineClassification(matrix):
	nearestNeighborCount = np.zeros	((20,20))
	
	for article in range(matrix.shape[0]):
		y = greatestSimiliarity(article, matrix)
		nearestNeighborCount[article/50, y/50] += 1
		# print('getting incremented: ', article/50, y/50)
		# print nearestNeighborCount
	print nearestNeighborCount
	averageClassificationError(nearestNeighborCount)
		



	

def makeHeatMap(data, names, color, outputFileName):
	#to catch "falling back to Agg" warning
	with warnings.catch_warnings():
   		warnings.simplefilter("ignore")
		#code source: http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
		fig, ax = plt.subplots()
		#create the map w/ color bar legend
		heatmap = ax.pcolor(data, cmap=color)
		cbar = plt.colorbar(heatmap)

		# put the major ticks at the middle of each cell
		ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
		ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

		# want a more natural, table-like display
		ax.invert_yaxis()
		ax.xaxis.tick_top()


		ax.set_xticklabels(range(1, 21))
		ax.set_yticklabels(names)

		plt.tight_layout()

		plt.savefig(outputFileName, format = 'png')
		plt.show()
		plt.close()


def dimensionReductionFunction(dimension):
	#matrix M 
	randList = []
	for i in range(dimension):
		rand = np.random.normal(0, 1.0, 61067)
		randList.append(list(rand))

	M = np.array(randList)
	reducedList = []
	for article in matrix:
		#multiply article transposed by M
		currArticle = article.toarray().T
		newVector = np.dot(M,currArticle)

		newVectorTransposed = newVector.T
		reducedList.append(newVectorTransposed)


	reducedMatrix = np.array(reducedList)
	reducedMatrix = np.reshape(reducedMatrix, (1000,dimension))
	print reducedMatrix

	#calculate cosine similarity between articles in reducedMatrix
	baselineClassification(reducedMatrix)


def averageClassificationError(matrix):
	averageErrors = 0
	for i in range(matrix.shape[0]):
		averageErrors += matrix[i,i]
	print averageErrors/float(1000)




read_data()
dimensionReductionFunction(10)