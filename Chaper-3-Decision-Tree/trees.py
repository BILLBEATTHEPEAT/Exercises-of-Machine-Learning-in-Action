#This is the implementation of Decision Tree copied from the book MLiA
#Author: Peter Harrington

from math import log

def clacShannonEnt(dataSet):      #This function will calculate the Shannon Entropy of the given dataSet
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]  	#assume that the labels are at the final column
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float( labelCounts[key] ) / numEntries	#the probability of every class
		shannonEnt -= prob * log(prob , 2)
	return shannonEnt

def createDataSet():
	dataSet = [[1,1,'yes'],
		[1,1,'yes'],
		[1,0,'no'],
		[0,1,'no'],
		[0,1,'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

def splitDataSet(dataSet , axis , value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)

	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = clacShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)                                           #to get the unique elements of the list
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet , i , value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob*clacShannonEnt(subDataSet)
		infoGain = baseEntropy -  newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return  bestFeature