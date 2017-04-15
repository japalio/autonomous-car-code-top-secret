import csv 
import numpy as np
from scipy.sparse import csr_matrix 
import math
import matplotlib.pyplot as plt
import warnings 

def createHashTables(dimension):
	global originalHash
	global M
	originalHash = [{}*128]
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
			if not key in originalHash[x]:
				originalHash[x][key] = articleId
			else:
				originalHash[x][key].append(articleId)

def calculateCosineSimilarity(aWordVector, bWordVector):
	aNorm = np.linalg.norm(aWordVector)
	bNorm = np.linalg.norm(bWordVector)
	dotProduct = np.dot(aWordVector, bWordVector.T)
	similarity = dotProduct / float((aNorm * bNorm))
	return similarity


def classification(dimension):
	global SqTotal
	global precisionCount
	global allSqs
	precisionCount = 0.0
	SqTotal = 0.0
	allSqs = []

	Sq = set()

	for articleId in range(orginalMatrix.shape[0]):
		for x in range(128):
			currentM = M[x]
			currArticle = originalMatrix[articleId].toarray().T
			newVector = np.dot(currentM,currArticle)

			key = np.where(newVector > 0, 1, 0)

			existingValues = orginalHash[x][key]
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












