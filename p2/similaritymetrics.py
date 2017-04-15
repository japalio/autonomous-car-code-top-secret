import csv 
import numpy as np
from scipy.sparse import csr_matrix 
import math
import matplotlib.pyplot as plt
import warnings 

#import csv file
#testing
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

def readGroupNames():
	reader = csv.reader(open('data/groups.csv','rb'))
	global groupNames 
	groupNames = []
	for row in reader:
		groupNames.append(row[0])
	return groupNames
		
def calculateCosineSimilarity(aWordVector, bWordVector):
	aNorm = np.linalg.norm(aWordVector.toarray())
	bNorm = np.linalg.norm(bWordVector.toarray())
	dotProduct = np.dot(aWordVector.toarray(), bWordVector.toarray().T)
	similarity = dotProduct / float((aNorm * bNorm))
	return similarity

def calculateJaccardSimilarity(aWordVector, bWordVector):
	return np.sum(np.minimum(aWordVector.toarray(),bWordVector.toarray()))/np.sum(np.maximum(aWordVector.toarray(), bWordVector.toarray()))

def calculateL2Similarity(aWordVector, bWordVector):

	return -1.0 * math.sqrt(np.sum(np.square(np.subtract(aWordVector.toarray(), bWordVector.toarray()))))


def similarity(a, articleAIndex, b, articleBIndex, similarityCode):

	#REMEMBER TO ADJUST FOR 0th INDEXING!!!

	#for news source a, grab that articleAIndexth word vector 
	#for news source b, grab the articleBIndexth word vector
	aWordVector = matrix[a * 50 + articleAIndex]
	bWordVector = matrix[b * 50 + articleBIndex] 


	if(similarityCode == 1):
		return calculateJaccardSimilarity(aWordVector, bWordVector)
	elif(similarityCode == 2):
		return calculateL2Similarity(aWordVector, bWordVector)
	else:
		#cosine similarity calculation:
		return calculateCosineSimilarity(aWordVector, bWordVector)
	
	
	#return the similarity between these two vectors


def compareArticles(a, b, similarityCode):
	averageSim = 0.0
	for k in range(0, 50):
		for l in range(0, k + 1):
			averageSim += similarity(a, k, b, l, similarityCode)
	averageSim *= 2
	averageSim /= float(2500.0) 
	return averageSim


def createPlotMatrix(similarityCode):
	plotMatrix = np.zeros((20, 20))
	for i in range(0, 20):
		for j in range(i + 1):
			averageSimilarity = compareArticles(i, j, similarityCode)
			plotMatrix[i][j] = averageSimilarity
			plotMatrix[j][i] = averageSimilarity

	# fig, ax = plt.subplots()
	# heatmap = ax.pcolor(plotMatrix, cmap=plt.cm.Blues, alpha=0.8)
	groupNames = readGroupNames()
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


#final output format:
#list of list of word Ids 
#list of list of counts 
#index of list in big list is the article ID, article ID 1-50 corresponds to group ID 1, 51-100 = group Id 2, etc.

#FML JK 
# https://piazza.com/class/j11oni3tp2f3wd?cid=55

#calculating similarity:
	#calculations between articles based on group #; 20x20 matrix based on average 

# def calculate_jaccard():
# 	#matrix of 20x20 

# 	#for news group in [1,20]:
# 		#for news group in [1,20]:
# 			#keep a running sum  = 0
# 			#for articles [1,50] in news group A:
# 				#for articles [1,50] in news group B:
# 					#calculate similarity between article from group A, article from group B
# 						#to calculate similarity: figure out words in common, divide by total # of words 
# 							#intersect word ids / union word ids 
# 					#add similarity to total running sum

# 			#avg similarity for group A and B = sum / (50*50)
# 			#update matrix with similarity 
# 			#also account FOR SYMMTETRY OF TABLE


# def calculate_l2():
# 	#http://stackoverflow.com/questions/16713368/calculate-euclidean-distance-between-two-vector-bag-of-words-in-python
# 	#basically that^ do we calculate based on intersection?

# def calculate_cosine(sparseMatrix):
# 	#use CSR matrix? 
# 	#dimensions = num articles x num words, where matrix[i][j] = article i's count for word j


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

	
def baselineClassification():
	nearestNeighborCount = np.zeros	((20,20))
	
	for article in range(matrix.shape[0]):
		y = greatestSimiliarity(article)
		nearestNeighborCount[article/50, y/50] += 1
		# print('getting incremented: ', article/50, y/50)
		# print nearestNeighborCount
	print nearestNeighborCount
		



	

def main():
	sparseMatrix = read_data()
	# calculate_jaccard()
	# calculate_l2()
	calculate_cosine()

read_data()
readGroupNames()

# createPlotMatrix(1)

baselineClassification()

# createPlotMatrix(1)




