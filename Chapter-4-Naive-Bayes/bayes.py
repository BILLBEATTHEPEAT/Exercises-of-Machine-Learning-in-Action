# MLiA chapter 4 Naive Bayes algorithm
# 2016.7.26

# The functions and codes in this file are mainly copied from the book Machine Learning in Action (MLiA) , author by Peter Harrington

# These are the functions used to prebatch necessary information

def loadDataSet():

	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
	['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
	['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
	['stop', 'posting', 'stupid', 'worthless', 'garbage'],
	['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
	['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]       #1 means junk words, 0 not
	return postingList , classVec

def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:	
		vocabSet = vocabSet | set(document)
	return ( vocabSet )

def setOfWords2Vec(vocabList , inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print "The word: %s is not in my Vocabulary!" % word
		return returnVec
