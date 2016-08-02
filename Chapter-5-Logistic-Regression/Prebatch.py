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

	return dataMat , labelMat

def sigmoid(inX):
	return 1.0 / ( 1 + np.exp(-inX) )

################################

#The following part of code is implemented by myself, which realized the Gradient Ascent Algorithm

def GraAscent(dataMatIn , classMatIn , rate , NumIterat):		#rate means the rate of learning, NumIterat means the time of iteration
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
