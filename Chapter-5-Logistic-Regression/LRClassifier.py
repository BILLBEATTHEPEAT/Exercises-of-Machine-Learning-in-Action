# Logistic Regression Classifier
#2016.8.3

import numpy as np
import Prebatch

def LRClassifier(inX , weights):
	prob = Prebatch.sigmoid( sum( inX * weights ) ) 
	if prob > 0.5 : return 1.0
	else : return 0.0

def test():
	dataArr , labelMat = Prebatch.loadDataSet()
	weights = Prebatch.StoGradAscent1(dataArr , labelMat , 0.01 , 10)
	right = 0.0
	numExam = dataArr.shape[0]
	for inx in range(dataArr.shape[0]):
		predict = LRClassifier(dataArr[inx][:3] , weights)
		if predict == labelMat[inx]:
			right += 1
		else:
			print"the unpredicted example:" , dataArr[inx]
			 	
	rate = right / float(numExam)
	print 'The rate of accuracy is' , rate



