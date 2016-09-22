# Realization of AdaBoost algorithm
# 2016.9.22

import numpy as np

def metaClassifierDS(dataMat , dimen , threshVal , threshIneq):
	retArr = np.ones(dataMat.shape[0] , 1)
	if threshIneq == 'lt' :
		retArr[dataMat[:,dimen] <= threshVal] = -1.0
	else:
		retArr[dataMat[:,dimen]  > threshVal] = -1.0

def adaboost(dataMat , labelClass , iterNum , numSteps):
	data = np.mat(dataMat)
	label = np.mat(labelClass).T

	numStep = float(numStep)
	bestStump = {}
	bestClasEst = mat( zeros( (data.shape[0] , 1) ) )
	minError = inf
	for i in range(n):
		rangeMin = data[: , i].min()
		rangeMax = data[:, i].max()
		step = int( (rangeMax - rangeMin) / numSteps )

		for j in range( -1 , step + 1):
			pass

			for sign in ['lt' , 'rt']:
				pass


	return bestStump , bestClasEst , minError
