import csv 
import numpy as np
from scipy.sparse import csr_matrix 
import math
import matplotlib.pyplot as plt
import warnings 

def read_data():
	reader = csv.reader(open('data/data50.csv', 'rb'))
	#matrix create here
	global originalMatrix
	originalMatrix = np.zeros((1000, 61067))
	for row in reader:
		articleid = int(row[0]) - 1
		wordid = int(row[1]) - 1
		wordcount = int(row[2])
		
		originalMatrix[articleid][wordid] = wordcount 
	newMatrix = np.zeros((1000, 61067))

	#NORMALIZE THE ORIGINAL MATRIX 
	for articleId in range(originalMatrix.shape[0]):
		#divide each article by its norm
		article = originalMatrix[articleId]
		norm = np.linalg.norm(article)
		newArticle = article / float(norm)
		newMatrix[articleId] = newArticle
	originalMatrix = newMatrix



def createHashTables(dimension):
	global originalHash
	global M
	originalHash = [dict() for x in range(128)]
	M = []

	for i in range(128):
		randList = []
		for i in range(dimension):
			rand = np.random.normal(0, 1.0, 61067)
			randList.append(list(rand))
		M.append(np.array(randList))

	for articleId in range(originalMatrix.shape[0]):
		for x in range(128):
			currentM = M[x]


			currArticle = originalMatrix[articleId].toarray().T
			newVector = np.dot(currentM,currArticle)

			key = np.where(newVector > 0, 1, 0)
			key = np.array_str(key)
			if not key in originalHash[x]:
				articleIdList = []
				articleIdList.append(articleId)
				originalHash[x][key] = articleIdList
			else:

				originalHash[x][key].append(articleId)

def calculateCosineSimilarity(aWordVector, bWordVector):

	#already normalized, just need dot product
	similarity = np.dot(aWordVector, bWordVector.T)
	return similarity


def classification(dimension):
	global nearestNeighborCount
	nearestNeighborCount = np.zeros	((20,20))
	global SqTotal
	global precisionCount
	global allSqs
	precisionCount = 0.0
	SqTotal = 0.0
	allSqs = []

	Sq = set()

	for articleId in range(originalMatrix.shape[0]):
		for x in range(128):
			currentM = M[x]
			currArticle = originalMatrix[articleId].toarray().T
			newVector = np.dot(currentM,currArticle)

			key = np.where(newVector > 0, 1, 0)
			key = np.array_str(key)
			existingValues = originalHash[x][key]
			for num in existingValues:
				if(num != articleId):
					Sq.add(num)

		SqTotal += len(Sq)
		

		maxSimilarity = 0.0
		maxArticle = None
		for article in Sq:
			#cosine similarity 
			similarity = calculateCosineSimilarity(originalMatrix[article].toarray(), originalMatrix[articleId].toarray())
			if(similarity > maxSimilarity):
				maxArticle = article
				maxSimilarity = similarity

		#found max by now
		if(maxArticle/50 == articleId/50):
			precisionCount += 1

		nearestNeighborCount[maxArticle/50, articleId/50] += 1
	print nearestNeighborCount



read_data()
createHashTables(5)
classification(5)







