import numpy as np  
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import collections 


#not sure if pca.mean is what we want here? 
def pca_recover():
	pcaX = PCA(n_components=1)
	pcaY = PCA(n_components=1)
	pcaY.fit(Y)
	Y_var = pcaY.mean_
	pcaX.fit(X)
	X_var = pcaX.mean_
	print Y_var/X_var





X = np.array([0.001 + 0.001*x for x in range(1000)])
X= X.reshape(-1,1)
Y = 2*X 
# print Y 

pca_recover()
