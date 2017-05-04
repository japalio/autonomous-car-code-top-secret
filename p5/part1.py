import numpy as np 

global words 

def read_data():
	global matrix 
	matrix = np.genfromtxt ('co_occur.csv', delimiter=",")
	return matrix 

def read_dictionary():
	text_file = open("dictionary.txt", "r")
	words = text_file.read().split('\n')


def normalizeMatrix():
	matrix += 1
	matrix = numpy.log(matrix)
	return matrix 


read_dictionary()
matrix = read_data()
matrix = normalizeMatrix(matrix)
