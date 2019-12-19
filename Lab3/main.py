import numpy as np
from generatedata import  get_dataset
from pcastudy import centralization , get_projection_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mean = np.array([1,2,3])
cov = np.array([[0.01,0,0],[0,10,0],[0,0,10]])
numbers = 30
dataset = get_dataset(mean,cov,numbers)
dataset = np.array(dataset)
dataset,meanvalue= centralization(np.transpose(dataset))
project_matrix = get_projection_matrix(dataset,2)
print(project_matrix)
new_dataset = project_matrix.dot(dataset)
print(new_dataset)
new_X = new_dataset[0]
new_Y = new_dataset[1]
X = dataset[0]
Y = dataset[1]
Z = dataset[2]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X,Y,Z,c='b')
ax.scatter(new_X,new_Y,c='r')
plt.show()