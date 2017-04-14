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
	print matrix.shape
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
	dotProduct = np.dot(aWordVector, bWordVector)
	similarity = dotProduct / float((aNorm * bNorm))
	return similarity



def similarity(a, articleAIndex, b, articleBIndex):
	#REMEMBER TO ADJUST FOR 0th INDEXING!!!


	#for news source a, grab that articleAIndexth word vector 
	#for news source b, grab the articleBIndexth word vector

	aWordVector = matrix[(a - 1) * 50]
	bWordVector = matrix[(b - 1) * 50] 

	#cosine similarity calculation:
	return calculateCosineSimilarity(aWordVector, bWordVector)
	
	
	#return the similarity between these two vectors

def compareArticles(a, b):
	averageSim = 0.0
	for k in range(0, 50):
		for l in range(0, k + 1):
			averageSim += similarity(a, k, b, l)
	averageSim *= 2
	averageSim /= 2500
	return averageSim




def createPlotMatrix():
	plotMatrix = np.zeros((20, 20))
	for i in range(0, 20):
		for j in range(i + 1):
			averageSimilarity = compareArticles(i, j)
			plotMatrix[i][j] = averageSimilarity
			plotMatrix[j][i] = averageSimilarity
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
createPlotMatrix()

