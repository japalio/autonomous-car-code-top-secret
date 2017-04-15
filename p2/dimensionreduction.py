import numpy as np
def dimensionReductionFunction(dimension):
	#matrix M 
	randList = []
	for i in range(dimension):
		rand = np.random.normal(0, 1.0, 10)
		randList.append(list(rand))

	
	M = np.array(randList)

	print M

dimensionReductionFunction(10)