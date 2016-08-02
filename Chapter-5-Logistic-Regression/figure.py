#This part of codes is simply copied from the book 
#`Coz I'm not so familiar  with the package matplotlib...
# Some of the codes have been corrected so that this function can fit the function that I have coded by myself.
#Authored by Peter Harrington
#2016.8.2

import numpy as np

def plotBestFit(wei):
	import matplotlib.pyplot as plt
	weights = wei 	#.getA() 		#wei is already in the form of np.array

	import Prebatch
	dataMat , labelMat = Prebatch.loadDataSet()
	dataArr = np.array(dataMat)
	n = np.shape(dataArr)[0]
	xcord1 = [] ; ycord1 = []
	xcord2 = [] ; ycord2 = []

	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i , 1]) ; ycord1.append(dataArr[i , 2])
		else:
			xcord2.append(dataArr[i , 1]) ; ycord2.append(dataArr[i , 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1 , ycord1 , s = 30 , c = 'red' , marker = 's')
	ax.scatter(xcord2 , ycord2 , s = 30 , c = 'green')
	x = np.arange(-3.0 , 3.0 , 0.1)
	y = (-weights[0] - weights[1]*x) / weights[2]
	ax.plot(x , y)
	plt.xlabel('x1') ; plt.ylabel('x2');
	plt.show()