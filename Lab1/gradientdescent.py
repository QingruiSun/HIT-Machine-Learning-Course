import numpy as np


class GradientDescent(object):
    def __init__(self, X, T, hyper, rate=0.03, delta=1e-6):
        """

        :param X: 训练数据自变量
        :param T: 训练数据函数值
        :param hyper: 惩罚参数
        :param rate: 迭代步长
        :param delta: 迭代误差值
        """
        self.X = X
        self.T = T
        self.hyper = hyper
        self.rate = rate
        self.delta = delta

    def lossfunc(self, w):
        #当得到系数w时，求此时训练结果与实际的误差值
        wt = np.transpose(w)
        temp = self.X @ w - self.T
        loss = 0.5 * (temp.T @ temp + self.hyper * w @ wt)
        return loss

    def derivativefunc(self, w):
        #求梯度
        return np.transpose(self.X) @ self.X @ w + self.hyper * w -\
         self.X.T @ self.T

    def fitting(self, w_0):
        #梯度下降法解方程
        loss0 = self.lossfunc(w_0)#系数为w时loss函数的值
        k = 0
        w = w_0
        while True:
            wt = w - self.rate * self.derivativefunc(w)#derivativefunc是求梯度的函数
            loss = self.lossfunc(wt)
            if np.abs(loss - loss0) < self.delta:
                break
            else:
                k = k + 1
                if loss > loss0:
                    self.rate *= 0.5#loss函数的值比前一次loss函数的值大，说明步长过大
                loss0 = loss
                w = wt
        return w