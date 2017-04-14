import csv 

wordIds = []
wordCounts = [] 
#import csv file
def read_data():
  reader = csv.reader(open('data/data50.csv', 'rb'))
  for row in reader:
  	print row 


#final output format:
#list of list of word Ids 
#list of list of counts 
#index of list in big list is the article ID, article ID 1-50 corresponds to group ID 1, 51-100 = group Id 2, etc.



#calculating similarity:
	#calculations between articles based on group #; 20x20 matrix based on average 

def calculate_jaccard():
	#matrix of 20x20 

	#for news group in [1,20]:
		#for news group in [1,20]:
			#keep a running sum  = 0
			#for articles [1,50] in news group A:
				#for articles [1,50] in news group B:
					#calculate similarity between article from group A, article from group B
						#to calculate similarity: figure out words in common, divide by total # of words 
							#intersect word ids / union word ids 
					#add similarity to total running sum

			#avg similarity for group A and B = sum / (50*50)
			#update matrix with similarity 
			#also account FOR SYMMTETRY OF TABLE


def calculate_l2():
	#http://stackoverflow.com/questions/16713368/calculate-euclidean-distance-between-two-vector-bag-of-words-in-python
	#basically that^ do we calculate based on intersection?

def calculate_cosine():
	#use CSR matrix? 
	#dimensions = num articles x num words, where matrix[i][j] = article i's count for word j


def main():
	wordIds, wordCounts = read_data()
	calculate_jaccard(wordIds, wordCounts)
	calculate_l2(wordIds, wordCounts)

