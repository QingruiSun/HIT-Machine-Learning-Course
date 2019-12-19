import numpy as np
import matplotlib.pyplot as plt
import random
co = ['r','g','b']

def get_dataset(k,numbers,means):
    dataset = []
    for i in range(0,k):
        tempnumber = numbers[i]
        tempmeans = means[i]
        tempcov = [[0.05,0],[0,0.05]]
        partdataset = np.random.multivariate_normal(tempmeans,tempcov,tempnumber)
        plt.scatter(partdataset[:, 0], partdataset[:, 1],c=co[i])
        dataset.append(partdataset)
    dataset = np.array(dataset)
    totalnumber = 0
    for i in range(k):
        totalnumber = totalnumber + numbers[i]
    dataset = dataset.reshape((totalnumber,2))
    for i in range(totalnumber):
        j = random.randint(0,totalnumber-1)
        temparray = dataset[j]
        dataset[j] = dataset[i]
        dataset[i] = temparray
    return dataset,totalnumber



