import numpy as np 
import math
from PIL import Image
import matplotlib.image as img
import scipy.misc
from matplotlib import pyplot as plt


#black pixel = 0
#white pixel = 1

def computeApproximation(k):

	#WE NEED TO CONVERT THE IMAGE TO BLACK AND WHITE PIXEL REPRESENTATION? 
	#OR can we assume it's already in that form...?
	# image = img.imread('p5_image.gif')
	
	im = Image.open("p5_image.gif")
	pix = im.load()
	#pix[x,y] prints the pixel at that spot!

	#why is this size transposed...? 1170 x 1160...
	print im.size

	u, s, v = np.linalg.svd(im, full_matrices=True, compute_uv=True)
	return s[:k]



#Note that the recovered drawing will have pixel values outside of the range [0, 1]; feel free to either scale things so that
#the smallest value in the matrix is black and the largest is white (default for most python packages
# and matlab), or to clip values to lie between 0 and 1.
def recoverDrawing(k):
	im = Image.open("p5_image.gif")
	#im = scipy.misc.imread("p5_image.gif")
	# pix = im.load()
	u, s, v = np.linalg.svd(im, full_matrices=True, compute_uv=True)
	
	#create s array with only the top k diagonal values
	S_k = np.diag(s[:k])

	#grab first k columns
	U_k = u[:,:k]

	#grab first k rows
	V_k = v[:k:]


	finalImageArray = np.matmul(np.matmul(U_k, S_k), V_k)
	finalImageArray = np.clip(finalImageArray, 0, 1)
	showImage(finalImageArray)


def showImage(array, imageName = None):
    img = Image.fromarray(array, "RGB")
    plt.imshow(array, cmap = 'gray')#, interpolation = 'nearest')
    plt.show()
 



recoverDrawing(1)
# print computeApproximation(1170)

# kValues = [1, 3, 10, 20, 50, 100, 150, 200, 400, 800, 1170]
# for kVal in kValues:
# 	rank = computeApproximation(kVal)
# 	print "K: ", kVal, " rank: ", rank


