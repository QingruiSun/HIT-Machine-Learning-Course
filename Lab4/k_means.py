import numpy as np
import matplotlib.pyplot as plt
import random
from datagenerator import get_dataset

#计算a,b两点的欧式距离
def calculate_distance(a,b):
    distance = (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])
    return distance

#k-means算法的迭代过程
def k_means_iteration(dataset,samplevector,k,totalnumber):
    lastvector = np.zeros((k,2))
    nowvector = samplevector
    iteravalue = 0
    for i in range(k):
        iteravalue = iteravalue + calculate_distance(lastvector[i],nowvector[i])
    iteratimes = 0
    classresult = 0
    while (iteratimes<100) and (iteravalue>0.00000001):
        iteratimes = iteratimes + 1
        classfier = []
        for i in range(totalnumber):
            flag = -1
            minvalue = 1000000
            for j in range(k):
                tempdistance = calculate_distance(dataset[i],nowvector[j])
                if tempdistance <= minvalue:
                   minvalue = tempdistance
                   flag = j
            classfier.append(flag)
        box = []
        for i in range(k):
            box.append([])
        for i in range(totalnumber):
            box[classfier[i]].append(i)
        box = np.array(box)
        classresult = box
        lastvector = nowvector
        nowvector = []
        for i in range(k):#对当前分类后的每一个类别的元素
            length = len(box[i])
            xvalue = 0
            yvalue = 0
            for j in range(length):#遍历某一类别的全部元素
                xvalue = dataset[box[i][j]][0] + xvalue
                yvalue = dataset[box[i][j]][0] + yvalue
            xvalue = xvalue/length
            yvalue = yvalue/length
            nowvector.append([xvalue,yvalue])
        iteravalue = 0
        for i in range(k):
            iteravalue = iteravalue + calculate_distance(lastvector[i], nowvector[i])
    print(nowvector)
    for i in range(k):
        x_value = []
        y_value = []
        length = len(classresult[i])
        for j in range(length):
            x_value.append(dataset[classresult[i][j]][0])
            y_value.append(dataset[classresult[i][j]][1])
        plt.scatter(x_value,y_value)
    nowvector = np.array(nowvector)
    plt.scatter(nowvector[:,0],nowvector[:,1],marker='+')
    plt.show()

k = 3
means = [[1,1],[1,6],[6,6]]
numbers = [100,100,100]
dataset,totalnumber= get_dataset(k,numbers,means)
samplevector = []
# for i in range(k):
#     tempnumber = random.randint(0,totalnumber-1)
#     samplevector.append(dataset[tempnumber])
samplevector.append([0.8,1.2])
samplevector.append(([1.2,5.6]))
samplevector.append([5.6,6.4])
k_means_iteration(dataset,samplevector,k,totalnumber)

