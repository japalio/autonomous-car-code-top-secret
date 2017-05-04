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