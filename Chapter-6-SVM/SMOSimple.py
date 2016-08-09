# The realization of SVM 
# Copied from the book MLiA, additionally edited with my commentary
#2016.8.9

from numpy import *
from Prebatch import *


def smoSimple( dataMatIn , classLabels , C , toler , maxIter):
	dataMatrix = mat(dataMatIn) ; labelMat = mat(classLabels).transpose()
	b = 0 ; m , n = shape(dataMatrix)
	alphas = mat(zeros( (m , 1) ))
	iter = 0
	while (iter < maxIter):		# The upper layer of iteration
		alphaPairsChanged = 0
		for i in range(m):	# The second layer of iteration
			fXi = float( multiply( alphas , labelMat ).T * ( dataMatIn * dataMatrix[i , :].T ) ) + b
			Ei = fXi - float( labelMat[i] )
			if ( (labelMat[i] * Ei < -toler) and ( alphas[i] < C ) ) or ( (labelMat[i] * Ei > toler) and (alphas[i] > 0) ):
				j = selectJrand(i , m)