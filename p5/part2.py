import numpy as np 
import math
from PIL import Image


#black pixel = 0
#white pixel = 1

def computeApproximation(k):

	#WE NEED TO CONVERT THE IMAGE TO BLACK AND WHITE PIXEL REPRESENTATION? 
	#OR can we assume it's already in that form...?
	img = Image.open('p5_image.gif')
	u, s, v = np.linalg.svd(img, full_matrices=True, compute_uv=True)
	return s[:k]



#Note that the recovered drawing will have pixel values outside of the range [0, 1]; feel free to either scale things so that
#the smallest value in the matrix is black and the largest is white (default for most python packages
# and matlab), or to clip values to lie between 0 and 1.
def recoverDrawing(150):
	return None


kValues = [1, 3, 10, 20, 50, 100, 150, 200, 400, 800, 1170]
for kVal in kValues:
	rank = computeApproximation(kVal)
	print "K: ", kVal, " rank: ", rank


