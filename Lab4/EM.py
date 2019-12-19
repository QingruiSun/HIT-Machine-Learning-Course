import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from datagenerator import get_dataset

def init_model_para(coefficient,meanvector,variancematrix):
    coefficient.append(0.2)
    coefficient.append(0.3)
    coefficient.append(0.5)
    meanvector.append([1.2, 1.1])
    meanvector.append([2.8, 3.2])
    meanvector.append([3, 4.9])
    variancematrix.append([[1.1, 0], [0, 0.9]])
    variancematrix.append([[1.05, 0], [0, 1.04]])
    variancematrix.append([[0.88, 0], [0, 1.12]])
    return coefficient,meanvector,variancematrix

def calculate_probability(covariancematrix,partmeanvector,l,sample):
    """
    计算在一特定参数分布下取得某一样本点sample的条件概率
    :param covariancematrix: 协方差矩阵
    :param partmeanvector: 属性均值向量
    :param l: 样本属性个数
    :param sample: 样本点向量
    :return: 条件概率值
    """
    covariancematrix = np.array(covariancematrix,'float32')
    partmeanvector = np.array(partmeanvector)
    sample = np.array(sample)
    tempmatrix = np.linalg.inv(covariancematrix)
    sample = sample - partmeanvector
    powerindedx = np.transpose(sample).dot(tempmatrix).dot(sample)
    tempa = math.pow(2*math.pi,l/2)
    tempb = np.linalg.det(covariancematrix)
    tempb = math.pow(tempb,0.5)
    value = (1/(tempa * tempb)) * math.pow(math.e,-0.5*powerindedx)
    return value

#计算每一个样本点属于某一个高斯混合模型的后验概率
def get_base_probability(totalnumber,dataset,variancematrix,meanvector,l):
    probilityvector = []  # 各样本属于各个混合成分的概率
    baseprobilityvector = []#在各个参数条件下，取得样本值的条件概率
    for i in range(totalnumber):
        partprobilityvector = []
        for j in range(k):
            covariancematrix = variancematrix[j]
            partmeanvector = meanvector[j]
            tempprobability = calculate_probability(covariancematrix, partmeanvector, l, dataset[i])
            partprobilityvector.append(tempprobability)
        baseprobilityvector.append(partprobilityvector)
    baseprobilityvector = np.array(baseprobilityvector)
    for j in range(totalnumber):
        tempa = 0
        tempvector = []
        for i in range(k):
            tempa = tempa + coefficient[i] * baseprobilityvector[j][i]
        for i in range(k):
            tempb = (coefficient[i] * baseprobilityvector[j][i]) / tempa
            tempvector.append(tempb)
        probilityvector.append(tempvector)
    return probilityvector


#M步，通过迭代更新参数
def calculate_new_parameter(coefficient,meanvector,variancematrix,dataset,totalnumber,probability,k,l):
    newcoefficient = []
    newmeanvector = []
    newvariancematrix = []
    #计算新的均值向量
    for i in range(k):
        tempa = 0
        for j in range(totalnumber):
            tempa = tempa + probability[j][i]
        tempvector = []
        for ux in range(l):
            tempvector.append(0)
        for j in range(totalnumber):
            for index in range(l):
                tempvector[index] = tempvector[index] + probability[j][i] * dataset[j][index]
        tempvector = tempvector / tempa
        newmeanvector.append(tempvector)
    #计算新的协方差矩阵
    for i in range(k):
        tempa = 0
        tempb = np.zeros((l,l))
        for j in range(totalnumber):
            tempa = tempa + probability[j][i]
        for j in range(totalnumber):
            tempc = np.matmul(np.array(dataset[j] - meanvector[i]).reshape((l,1)),np.transpose(np.array(dataset[j] - meanvector[i])).reshape((1,l)))
            tempb = tempb + probability[j][i] * tempc
        tempb = tempb / tempa
        newvariancematrix.append(tempb)
    #计算新的系数向量
    for i in range(k):
        tempa = 0
        for j in range(totalnumber):
            tempa = tempa + probability[j][i]
        tempa = tempa / totalnumber
        newcoefficient.append(tempa)
    return newcoefficient,newmeanvector,newvariancematrix

#计算似然值
def calculate_likely_value(dataset,coefficient,meanvector,variancematrix,k,l,totalnumber):
    tempb= 0
    for j in range(totalnumber):
        tempa = 0
        for i in range(k):
            tempa = tempa + coefficient[i] * calculate_probability(variancematrix[i],meanvector[i],l,dataset[j])
        tempa = math.log(tempa)
        tempb = tempb + tempa
    return tempb

#EM算法函数，实现EM算法核心步骤
def EM_algorithm(coefficient,meanvector,variancematrix,dataset,totalnumber,k,l):
    looptimes = 0
    prob = []
    while looptimes < 15:
        probilityvector = get_base_probability(totalnumber, dataset, variancematrix, meanvector, l)
        coefficient, meanvector, variancematrix = calculate_new_parameter(coefficient, meanvector, variancematrix,
                                                                          dataset, totalnumber, probilityvector, k, l)
        likelyvalue = calculate_likely_value(dataset,coefficient,meanvector,variancematrix,k,l,totalnumber)
        prob = probilityvector
        print("likelyvalue")
        print(likelyvalue)
        """
        print(coefficient)
        print(meanvector)
        print(variancematrix)
        """
        looptimes = looptimes + 1
    meanvector = np.array(meanvector)
    classfier = []
    for i in range(totalnumber):
        maxnum = 0
        flagval = -1
        for j in range(k):
            if prob[i][j] > maxnum:
                maxnum = prob[i][j]
                flagval = j
        classfier.append(flagval)
    #plt.scatter(dataset[:,0],dataset[:,1])
    plt.scatter(meanvector[:,0],meanvector[:,1],marker='+')
    plt.show()
    return classfier


k = 3#高斯混合成分的个数
l = 2#每个样本属性的个数
means = [[1,1],[0.5,1.5],[1,2]]
numbers = [100,100,100]
dataset,totalnumber = get_dataset(k,numbers,means)
coefficient = []#高斯混合模型的系数
meanvector = []#高斯混合模型的均值参数
variancematrix = []#高斯混合模型的方差参数
coefficient,meanvector,variancematrix = init_model_para(coefficient,meanvector,variancematrix)
EM_algorithm(coefficient,meanvector,variancematrix,dataset,totalnumber,k,l)
"""
data = pd.read_csv('iris.data')
data = np.array(data)
data = np.delete(data,4,axis=1)
meanvector = [[4,3,1.5,0.2],[6,3,5,2],[6,3,4,1]]
coefficient = [0.2,0.3,0.5]
variancematrix = [[[0.4,0,0,0],[0,0.15,0,0],[0,0,0.06,0],[0,0,0,0.03]],[[0.12,0,0,0],[0,0.21,0,0],[0,0,0.13,0],[0,0,0,0.12]],
                  [[0.06,0,0,0],[0,0.56,0,0],[0,0,0.05,0],[0,0,0,0.74]]]
classifier = EM_algorithm(coefficient,meanvector,variancematrix,data,149,3,4)
print(classifier)
data = pd.read_csv('iris.data')
data = np.array(data)
irisres = data[:,4]
numbers = 0
for i in range(149):
    if irisres[i] == 'Iris-setosa':
        irisres[i] = 0
    if irisres[i] == 'Iris-versicolor':
        irisres[i] = 2
    if irisres[i] == 'Iris-virginica':
        irisres[i] = 1
for i in range(149):
    if irisres[i] == classifier[i]:
        numbers = numbers + 1
print(irisres)
print(numbers)
"""






