# This code is the implementation of the simple Naive Bayes classifier author by myself
# 2016.7.27

# Powered by Micheal Ricardo Huang

import numpy as np
from bayes import *

def TrainTheClassifier(trainMatrix , trainLabel):	#This function will return 3 arg, the first one is p(c), the second one is p(w|1), the third one is p(w2|0)
						#assume that the input arg trainMatrix is a vector which trainsformed by the function createVocabList
	
	NumExams = len(trainMatrix)		# number of the examples (email, text or some file like that)
	NumWords = len(trainMatrix[0])		# number of the words of the vocabulary list of every example

	p1 = ones(NumWords)			#p1 means the number of true aimed examples,
	p0 = ones(NumWords)			#p2 means the number of the opposite  examples

	p1num = 2.0
	p0num = 2.0

	for i in range(NumExams):
		if trainLabel[i] == 1:
			p1 += trainMatrix[i]
			p1 += sum(trainMatrix[i])
		else:
			p0 += rainMatrix[i]
			p0 += sum(trainMatrix[i])

	p_w_1= log(p1 / p1num)
	p_w_0 = log(p0 / p0num)
	p_c = log(sum(trainLabel) / NumExams)
	p_not_c =  log(1 - sum(trainLabel) / NumExams)

	return p_w_0 , p_w_1 , p_c 

def classifierNB(vec2Classify , p0Vec , p1Vec , pClass1 , pClass0):
	p1 = sum(vec2Classify * p1Vec) + pClass1
	p0 = sum(vec2Classify * p0Vec) + pClass0

	if p1 > p0:
		return 1
	elif p1 < p0:
		return 0
	else:
		print "This can be regard as any type as you like"
		return -1
