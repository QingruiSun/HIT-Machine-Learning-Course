import numpy as np


class AnalyticalSolution(object):
    def __init__(self, X, T):
        self.X = X
        self.T = T

    def fitting(self):
        #不带惩罚项的解析解
        temp = np.transpose(self.X) @ self.X
        res = np.linalg.inv(temp) @ np.transpose(self.X) @ self.T
        return res

    def fitting_with_regulation(self, hyper):
        #带惩罚项的解析解
        return np.linalg.solve(np.identity(len(self.X.T)) * hyper + \
        self.X.T @ self.X, self.X.T @ self.T)