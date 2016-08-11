# The realization of SVM 
# Copied from the book MLiA, additionally edited with my commentary
#2016.8.9

from numpy import *
from Prebatch import *


def smoSimple( dataMatIn , classLabels , C , toler , maxIter):  
				# this is a simple demo of the SMO algorithm
				# random choosing alpha pair to optimi
				# C means a constant which is used to measure the distance of support vector
				# toler means the tolerance of making mistakes
				#maxIter means the maximun number of iteration

	dataMatrix = mat(dataMatIn) ; labelMat = mat(classLabels).transpose()  #make labelMat a vector
	b = 0 ; m , n = shape(dataMatrix)
	alphas = mat(zeros( (m , 1) ))
	iter = 0					
	while (iter < maxIter):		# The upper layer of iteration
		alphaPairsChanged = 0	#showing whether the alpha pair have been changed during a round

		for i in range(m):	# The second layer of iteration
			fXi = float( multiply( alphas , labelMat ).T * ( dataMatIn * dataMatrix[i , :].T ) ) + b
				# the prediction of example i
			Ei = fXi - float( labelMat[i] )	#error of prediciton

			if ( (labelMat[i] * Ei < -toler) and ( alphas[i] < C ) ) or ( (labelMat[i] * Ei > toler) and (alphas[i] > 0) ):
							#both the positive and negtive interval wiil be measured

				j = selectJrand(i , m)	# choose the other alpha randomly

				fXj = float( multiply( alphas , labelMat ).T * (dataMatrix * dataMatrix[j,:].T) ) + b
				Ej = fXj - float( labelMat[j] )
							#calculate the prediciton and error
				alphaIold = alphas[i].copy()
				alphaJold = alphas[j].copy()
						# allocate new memory for old value
				if (labelMat[i] != labelMat[j]):			# this part is using for making sure the value of alpha is between 0 and C
					L = max ( 0 , alphas[j] - alphas[i] )
					H = min ( C , C + alphas[j] - alphas[i] )
				else:
					L = max ( 0 , alphas[j] + alphas[i] - C)
					H = min ( C , alphas[j] + alphas[i] )

				if L == H:
					print "L = H" ; continue

				eta = 2.0 * dataMatrix[i , :] * dataMatrix[j , :].T - dataMatrix[j , :] * dataMatrix[j , :].T
							#the best modified quantity
				if eta >= 0:
					print "eta >= 0" ; continue  #assure that the funciton will become convergent

				alphas[j] -= labelMat[j] * (Ei - Ej) / eta
				alphas[j] = clipAlpha(alphas[j] , H , L)

				if ( abs( alphas[j] - alphaJold ) < 0.00001 ):			#exame that whether the change is too small
					print "j not moving enough" ; continue
				
				alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])     # alpha_i and alpha_j have been changed in opposite way

				b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i , :] * dataMatrix[i , :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i , :] * dataMatrix[j , :].T
				b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i , :] * dataMatrix[j , :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j , :] * dataMatrix[j , :].T
					
				if ( 0 < alphas[i] ) and ( C > alphas[j] ) :
					b = b1
				elif (0 < alphas[j]) and ( C > alphas[j] ):
					b = b2
				else:
					b = (b1 + b2) / 2				# update the constant b

				alphaPairsChanged += 1
				print "iter : %d i: %d, pairs changed %d" % (iter , i , alphaPairsChanged)

		if (alphaPairsChanged == 0):
			iter += 1
		else:
			iter = 0

		print "iteration number : %d " % iter

	return b , alphas
