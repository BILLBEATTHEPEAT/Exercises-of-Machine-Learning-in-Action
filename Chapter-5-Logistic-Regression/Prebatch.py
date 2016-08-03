#Implementation of the algorithm of the book MLiA

#Logistic Regression
#2016/8/1
#Micheal Huang

##############################
#This part is copied from the book, which will be used to prebatch some data for the algorithm

import re
import numpy as np

def loadDataSet():
	dataMat = []
	labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()       #remove the space from the string and seperate it into words

		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		#transform the input data into useful features of examples
		#the first feature '1.0' is used for drawing the decision boundary

		labelMat.append(int(lineArr[2]))

	return np.array(dataMat) , np.array(labelMat)

def sigmoid(inX):
	return 1.0 / ( 1 + np.exp(-inX) )

################################

#The following part of code is implemented by myself, which realized the Gradient Ascent Algorithm

def GradAscent(dataMatIn , classMatIn , rate , NumIterat):		#rate means the rate of learning, NumIterat means the time of iteration
	dataMat = np.array(dataMatIn)
	labelMat = np.array(classMatIn)
	weiMat = np.ones(dataMat.shape[1])
	#print weiMat.shape
	#print labelMat.shape
	#print dataMat.shape

	for time in range(NumIterat):
		y = sigmoid(dataMat.dot(weiMat))
		#print y.shape
		err = labelMat - y
		weiMat = weiMat + rate * (dataMat.T).dot(err) #here, the transpose method will transpose the matrix 
								      #which shows the same result as calculate the partial derivative
	return weiMat
	  
			######     REMEMBER: here the parameters and values that we use to calculate are all in the form of MATRIX!!!
			######                               So I have to use the method ".dot()"



#The following part is one of the simple version of  Stochastic Gradient Ascent Algorithm
#Finished by myself 
#2016.8.3
#version 0
def StoGradAscent0(dataMatIn , classMatIn , rate , NumIterat):
	dataMat = np.array(dataMatIn)
	labelMat = np.array(classMatIn)
	weiMat = np.ones(dataMat.shape[1])
	count = 0

	for time in range(NumIterat):

		for inx in range(dataMat.shape[0]):
			y = sigmoid( dataMat[inx].dot(weiMat) )
			err = labelMat[inx] - y
			weiMat = weiMat + rate * (dataMat[inx].T).dot(err)
		count += 1

	print 'Iterated %d times' % count
	return weiMat

	#This version simply iterates every examples in the dataMatIn to update the weights.
	# So it is not so precise as the former version (Non-stochastic version) in low iterate times.
	# As the result of my experiments, when the NumIterat is up to 100, this algorithm will show a good result.


#This is a better version of SGA
#completed by myself
#2016.8.3
#version 1
def StoGradAscent1(dataMatIn , classMatIn , rate , NumIterat):
	dataMat = np.array(dataMatIn)
	labelMat = np.array(classMatIn)
	alpha = rate

	weiMat = np.ones(dataMat.shape[1])
	count = 0

	for time in range(NumIterat):

		for i in range(dataMatIn.shape[0]):
			
			########  This function is copied from book MLiA
			alpha = 4/(1.0 + time + i) + 0.01
			########

			inx = np.random.randint(0 , dataMat.shape[0])
			y = sigmoid( dataMat[inx].dot(weiMat) )
			err = labelMat[inx] - y
			weiMat = weiMat + alpha * (dataMat[inx].T).dot(err)
			np.delete(dataMat , inx , axis = 0)
		count += 1

	print 'Iterated %d times' % count
	return weiMat



