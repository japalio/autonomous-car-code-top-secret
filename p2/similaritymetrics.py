import csv 
import numpy as np
from scipy.sparse import csr_matrix 

#import csv file
def read_data():
	reader = csv.reader(open('data/data50.csv', 'rb'))
	#matrix create here
	# matrix = csr_matrix((1000, 61067))
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
	aNorm = np.linalg.norm(aWordVector.toarray())
	bNorm = np.linalg.norm(bWordVector.toarray())
	dotProduct = np.dot(aWordVector.toarray(), bWordVector.toarray().T)
	similarity = dotProduct / float((aNorm * bNorm))
	# if isSame:
	# print similarity
	return similarity

def calculateJaccardSimilarity(aWordVector, bWordVector):
	return None

def calculateL2Similarity(aWordVector, bWordVector):
	return np.sum(np.square(np.subtract(aWordVector, bWordVector)))

def similarity(a, articleAIndex, b, articleBIndex, similarityCode):
	#REMEMBER TO ADJUST FOR 0th INDEXING!!!


	#for news source a, grab that articleAIndexth word vector 
	#for news source b, grab the articleBIndexth word vector

	aWordVector = matrix[a * 50 + articleAIndex]
	bWordVector = matrix[b * 50 + articleBIndex] 

	if(similarityCode == 1):
		return calculateJaccardSimilarity(aWordVector, bWordVector)
	elif(similarityCode == 2):
		return calculateL2Similiarity(aWordVector, bWordVector)
	else:
		#cosine similarity calculation:
		return calculateCosineSimilarity(aWordVector, bWordVector)
	
	
	#return the similarity between these two vectors

def compareArticles(a, b, similarityCode):
	averageSim = 0.0
	for k in range(0, 50):
		for l in range(0, k + 1):
			# print('k: ', k, ' l: ', l)
			averageSim += similarity(a, k, b, l, similarityCode)

	# if (a == b):
	# 	print ('in compare articles')
	# 	print averageSim
	averageSim *= 2
	averageSim /= 2500.0
	return averageSim




def createPlotMatrix(similarityCode):
	plotMatrix = np.zeros((20, 20))
	for i in range(0, 20):
		for j in range(i + 1):
			averageSimilarity = compareArticles(i, j, similarityCode)

			plotMatrix[i][j] = averageSimilarity
			plotMatrix[j][i] = averageSimilarity
	print plotMatrix
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

	

	

def main():
	sparseMatrix = read_data()
	# calculate_jaccard()
	# calculate_l2()
	calculate_cosine()

read_data()
print('hi')
createPlotMatrix(3)

