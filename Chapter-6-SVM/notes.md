#Notes of SVM

  > ##### Here are some notes about SVM taken down druing study. Parts of them are urls of some ingenious blogs published by Chinese authors.
  
###1. Dual variable Optimization (Lagrange Duality):
#####Reference:
  > http://blog.pluskid.org/?p=682
  
  
###2. Kernel Trick:
#####Reference:
  > http://blog.csdn.net/zouxy09/article/details/17291805
  
##### Two main tasks:
- Firstly, transfer whole primate information x to another feature space using unlinear projection  ![equation](http://latex.codecogs.com/gif.latex?\\Phi(x))
- Exame the classifier in the new feature space

##### Solution:
- For the first task, we can use Cover Theorem----project the feature to a high-dimension space. And according to the characteristic of support vector, we only need to calculate the inner product of the input samples and support vectors.
- Most frequently used function: Radial-basis function(RBF):  
![equation](http://latex.codecogs.com/gif.latex?\K(x,y) = e^{(-||x - y||^2/(2\\sigma^2))})

##### Comment(from reference):
  >支持向量机的决策过程也可以看做一种相似性比较的过程。首先，输入样本与一系列模板样本进行相似性比较，模板样本就是训练过程决定的支持向量，而采用的相似性度量就是核函数。样本与各支持向量比较后的得分进行加权后求和，权值就是训练时得到的各支持向量的系数αi和类别标号的成绩。最后根据加权求和值大小来进行决策。而采用不同的核函数，就相当于采用不同的相似度的衡量方法。
