import numpy as np  
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import collections 

def pca_recover():
	pca = PCA(n_components=1)
	pca.fit(Y)
	Y_var = pca.explained_variance_ 
	pca.fit(X)
	X_var = pca.explained_variance_
	print Y_var/X_var

	# print pca.fit(Y)/pca.fit(X)
	# print(pca.explained_variance_ratio_) 



X = np.array([0.001 + 0.001*x for x in range(1000)])
X= X.reshape(-1,1)
Y = 2*X 
# print Y 

pca_recover()
