import numpy as np  
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import collections 
from numpy.random import randn
import math 


#not sure if pca.mean is what we want here? 
def pca_recover(X1, Y1):
	matrix = np.hstack((X1,Y1))
	# pcaX = PCA(n_components=1)
	# pcaY = PCA(n_components=1)
	# print pcaX
	pcaXY = PCA(n_components=1)
	# print pcaXY.components_	
	# print XY.shape
	xy2 = pcaXY.fit(matrix)
	# print xy2.explained_variance_
	# XYfit = pcaXY.fit_transform(matrix)
	# print xy2.components_
	return xy2.components_[0][1]/xy2.components_[0][0]
	# v = matrix/XYfit
	# print v[0][1]/v[0][0]
	# return v[0][1]/v[0][0]

def ls_recover(X1, Y1):
	numerator = np.dot((X1-np.mean(X1)).T, Y1-np.mean(Y1))
	denominator = np.square(np.linalg.norm(X1 - np.mean(X1)))
	# print numerator/denominator
	ls = numerator/denominator
	return ls 

def noisyY():
	cvalues = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, .45, .5]
	pcaRecover = []
	lsRecover = []
	for c in cvalues:
		for i in range(30):
			noise = randn(1000)*math.sqrt(c)
			Ynew = [(2*x)/float(1000) for x in range(0,1000)]
			Yhat = noise + Ynew
			Yhat = Yhat.reshape(-1,1)
			pcaRecover.append(pca_recover(X, Yhat))
			lsRecover.append(ls_recover(X, Yhat))
			# print 'ls Recover ', ls_recover(X, Yhat), 'for c: ', c
			# print 'pca recover', pca_recover(X, Yhat), 'for c:', c

	#plot
	for pca in range(len(pcaRecover)):
		indexOfC = pca/30	
		
		cvalue = cvalues[indexOfC]
		plt.scatter(cvalue, pcaRecover[pca], label='PCA recover', color='red')
		plt.scatter(cvalue, lsRecover[pca], label='LS recover', color='blue')

	# plt.legend(loc = 'lower right')
	plt.show()

def noisyXandY():
	cvalues = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, .45, .5]
	pcaRecover = []
	lsRecover = []
	for c in cvalues:
		for i in range(30):
			noise = randn(1000)*math.sqrt(c)
			noiseForX = randn(1000)*math.sqrt(c)
			Xnew = X + noiseForX
			Xnew.reshape(-1,1)
			Ynew = [(2*x)/float(1000) for x in range(0,1000)]
			Yhat = noise + Ynew
			Yhat = Yhat.reshape(-1,1)
			pcaRecover.append(pca_recover(Xnew, Yhat))
			lsRecover.append((ls_recover(Xnew, Yhat))[0])
			# print 'ls Recover ', ls_recover(X, Yhat), 'for c: ', c
			# print 'pca recover', pca_recover(X, Yhat), 'for c:', c

	#plot

	# print lsRecover
	print len(pcaRecover)
	for pca in range(len(pcaRecover)):
		indexOfC = pca/30	
		
		cvalue = cvalues[indexOfC]
		plt.scatter(cvalue, pcaRecover[pca], label='PCA recover', color='red')
	
		plt.scatter(cvalue, lsRecover[pca], label='LS recover', color='blue')

	# plt.legend(loc = 'lower right')
	plt.show()






X = np.array([0.001 + 0.001*x for x in range(1000)])
X= X.reshape(-1,1)
# Y = 2*X 
# XY = np.hstack((X, Y))

# print Y 

noisyXandY()
# noisyY()