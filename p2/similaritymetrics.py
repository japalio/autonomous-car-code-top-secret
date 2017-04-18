import csv 
import numpy as np
from scipy.sparse import csr_matrix 
import math
import matplotlib.pyplot as plt
import warnings 

#import csv file, read into global variable 'matrix'
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

#read group  names into global variable 'groupNames'
def readGroupNames():
	reader = csv.reader(open('data/groups.csv','rb'))
	global groupNames 
	groupNames = []
	for row in reader:
		groupNames.append(row[0])
	return groupNames
		
#given two bag of word vectors in CSR format, calculate cosine similarity 
def calculateCosineSimilarity(aWordVector, bWordVector):
	aNorm = np.linalg.norm(aWordVector.toarray())
	bNorm = np.linalg.norm(bWordVector.toarray())
	dotProduct = np.dot(aWordVector.toarray(), bWordVector.toarray().T)
	similarity = dotProduct / float((aNorm * bNorm))
	return similarity

#given two bag of word vectors in CSR format, calculate Jaccard similarity 
def calculateJaccardSimilarity(aWordVector, bWordVector):
	return np.sum(np.minimum(aWordVector.toarray(),bWordVector.toarray()))/np.sum(np.maximum(aWordVector.toarray(), bWordVector.toarray()))

#given two bag of word vectors in CSR format, calculate L2 similarity 
def calculateL2Similarity(aWordVector, bWordVector):
	return -1.0 * math.sqrt(np.sum(np.square(np.subtract(aWordVector.toarray(), bWordVector.toarray()))))


#given which groups and the index of the articles within the groups, return similarity based on similarity code 
def similarity(a, articleAIndex, b, articleBIndex, similarityCode):
	#for news source a, grab that articleAIndexth word vector 
	#for news source b, grab the articleBIndexth word vector
	aWordVector = matrix[a * 50 + articleAIndex]
	bWordVector = matrix[b * 50 + articleBIndex] 


	if(similarityCode == 1):
		return calculateJaccardSimilarity(aWordVector, bWordVector)
	elif(similarityCode == 2):
		return calculateL2Similarity(aWordVector, bWordVector)
	else:
		return calculateCosineSimilarity(aWordVector, bWordVector)

#compare all articles between groups a and b based on similarity metric 
def compareArticles(a, b, similarityCode):
	averageSim = 0.0
	for k in range(0, 50):
		for l in range(0, 50):
			averageSim += similarity(a, k, b, l, similarityCode)
	# averageSim *= 2
	averageSim /= float(2500.0) 
	return averageSim

#create 20x20 plot, where each entry corresponds to average similarity over all ways of pairing up one article from A with one article from B.
def createPlotMatrix(similarityCode):
	plotMatrix = np.zeros((20, 20))
	for i in range(0, 20):
		for j in range(i + 1):
			averageSimilarity = compareArticles(i, j, similarityCode)
			plotMatrix[i][j] = averageSimilarity
			plotMatrix[j][i] = averageSimilarity
	makeHeatMap(plotMatrix, groupNames, plt.cm.Blues, 'heatMap1')

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


#used for baseline calculation, find the article with the max cosine similarity score to the current article 
def greatestSimiliarity(hostArticle):
	currMax = 0.0
	maxArticle = None
	for article in range(matrix.shape[0]):
		if(hostArticle != article):
			similarity = calculateCosineSimilarity(matrix[hostArticle], matrix[article])
			if(similarity > currMax):
				currMax = similarity
				maxArticle = article
	return maxArticle

	
#used to compare cosine similarity scores between every article
def baselineClassification():

	nearestNeighborCount = np.zeros	((20,20))
	for article in range(matrix.shape[0]):
		y = greatestSimiliarity(article)
		nearestNeighborCount[article/50, y/50] += 1

	# makeHeatMap(nearestNeighborCount, groupNames, plt.cm.Blues, 'heatMap1')
	averageClassificationPrecision(nearestNeighborCount)

		

def averageClassificationPrecision(m):
	averageErrors = 0
	for i in range(m.shape[0]):
		averageErrors += m[i,i]
	print averageErrors/float(1000)

read_data()
readGroupNames()
# baselineClassification()

createPlotMatrix(3)




