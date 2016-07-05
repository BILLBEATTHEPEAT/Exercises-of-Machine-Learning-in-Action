import numpy as np
import  operator

def classify0( inX , dataSet , labels , k ):
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile( inX , (dataSetSize , 1) ) - dataSet
	sqDiffMat = diffMat**2
	sqDIstances = sqDiffMat.sum(axis = 1)
	distances = sqDIstances**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range (k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel , 0) + 1
	sortedClassCount = sorted(classCount.iteritems(),
		key = operator.itemgetter(1) , reverse = True)
	return sortedClassCount[0][0]


