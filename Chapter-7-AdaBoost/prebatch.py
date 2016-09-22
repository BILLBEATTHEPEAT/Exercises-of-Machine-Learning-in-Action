# Data prebatching for the ealization of AdaBoost algorithm, inspired by MLiA
# 2016.9.22

def loadSimpleData():
	datMat = matrix([[1. , 2.1],
		[2. , 1.1],
		[1.3 , 1.],
		[1. , 1.],
		[2. , 1.]])
	labelMat = [1.0 , 1.0 , -1.0 , -1.0 , -1.0]
	return datMat , labelMat

def loadTestData():
	pass