import numpy as np

def centralization(X):
    """
    中心化X
    :param X: m*n矩阵,m代表样本个数，n代表一个样本点的维数
    :return: 中心化后的矩阵
    """
    rownumber = X.shape[0]#样本个数
    colnumber = X.shape[1]#每个样本点的维数
    meanvalue = []
    newdataset = []
    for i in range(0,colnumber):
        temp = 0
        for j in range(0,rownumber):
            temp = temp + X[j][i]
        meanvalue.append(temp/rownumber)
    for i in range(0,colnumber):
        tempdataset = []
        for j in range(0,rownumber):
            tempnumber = X[j][i] - meanvalue[i]
            tempdataset.append(tempnumber)
        newdataset.append(np.array(tempdataset))
    newdataset = np.transpose(np.array(newdataset))
    return newdataset,meanvalue

def get_projection_matrix(X,k):
    """
     PCA降维，获得投影矩阵
    :param X: n*m矩阵，n代表一个样本点数据的维数，m代表样本点的个数
    :param k: 降维后样本点的维数
    :return: 用来降维的投影矩阵
    """
    Q = X.dot(np.transpose(X))
    eigen_vals,eigen_vecs = np.linalg.eig(Q)
    sorted_indices = np.argsort(eigen_vals)
    eigen_vals = eigen_vals[sorted_indices[:-k-1:-1]]
    eigen_vecs = eigen_vecs[:,sorted_indices[:-k-1:-1]]
    eigen_vecs_trans = np.transpose(eigen_vecs)
    eigen_matrix = eigen_vecs_trans.dot(Q).dot(eigen_vecs)
    return eigen_vecs_trans
