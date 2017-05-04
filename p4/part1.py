import numpy as np  
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import collections 


#reading the data 
data = np.loadtxt('p4dataset2017.txt', dtype='string')
populations = data[:,2]
sex = data[:,1]
data = np.delete(data, 0, axis=1)
data = np.delete(data, 0, axis =1)
data = np.delete(data, 0, axis =1)

modes = stats.mode(data.T, axis=1)[0]

print type(modes)
matrix = np.zeros((995, 10101))
counter = 0
for row in data:
	matrixRow = np.array(row == modes.T, dtype=int) 
	matrix[counter] = matrixRow
	counter += 1
#invert because true and false values are inverted rn  
X = 1 - matrix
	


def runPCA(numComponents):
	pca = PCA(n_components=numComponents)
	pca.fit(X)
	print(pca.explained_variance_ratio_) 

	pca.n_components = numComponents
	X_reduced = pca.fit_transform(X)
	print X_reduced.shape

	XX = np.dot(X.T, X_reduced)
	# print XX.shape
	# print XX 
	# print XX[:,2]
	# print XX[:,2].argmax(axis=0)
	XXsort = np.argsort(XX[:,2])
	print XXsort[10100]
	print XXsort[10099]
	print XXsort[10098]
	# print XX[:,2][9696]
	# print XX[:,2][9960]
	# print XX[:,2][9614]

	# print XX[:,2][9912]


	
	#CHANGE THIS WHEN NUMCOMPONENTS IS CHANGED 
	# plotTwoComponents(X_reduced[:,0], X_reduced[:,numComponents-1])
	# plotGender(X_reduced[:,0], X_reduced[:,numComponents-1])

def plotTwoComponents(p1, p2):
	labelNames = ['ACB', 'ASW', 'GWD', 'YRI', 'LWK', 'MSL', 'ESN']
	lists = [[] for _ in range(14)]

	for i in range(995):
		index = labelNames.index(populations[i])
		index *= 2
		p1point = p1[i]
		p2point = p2[i]
		lists[index].append(p1point)
		lists[index + 1].append(p2point)

	labelIndex = 0
	colors = ['red', 'black', 'blue', 'brown', 'green', 'yellow', 'orange']
	for i in range(0,14,2):
		# print i 
		print len(lists[i])
		print len(lists[i+1])
		plt.scatter(lists[i], lists[i+1], label = labelNames[labelIndex], color=colors[labelIndex])
	
		labelIndex += 1

	plt.legend(loc = 'lower right')
	plt.show()


def plotGender(p1, p2):
	labelNames = ['1', '2']
	labelNamesString = ['male', 'female']
	lists = [[] for _ in range(4)]

	for i in range(995):
		index = labelNames.index(sex[i])
		index *= 2
		p1point = p1[i]
		p2point = p2[i]
		lists[index].append(p1point)
		lists[index + 1].append(p2point)

	labelIndex = 0
	colors = ['red', 'blue']
	for i in range(0,4,2):
		# print i 
		plt.scatter(lists[i], lists[i+1], label = labelNamesString[labelIndex], color=colors[labelIndex])
	
		labelIndex += 1

	plt.legend(loc = 'lower right')
	plt.show()


runPCA(3)

