import numpy as  np
import math
import matplotlib.pyplot as plt

class calculate_neww:
    def __init__(self,m,n,X,Y):
        self.m = m
        self.n = n
        self.X = X
        self.Y = Y
        self.w = np.ones((n,1))
        self.wd = np.zeros((n,1))
    def differentialfunc(self):#求损失函数的导数
        tempa = np.zeros((self.m,1))
        for i in range(0,self.m):
            for j in range(0,self.n):
                tempa[i] = self.X[i][j]*self.w[j] + tempa[i]
        for j in range(0,self.n):
            tempb = 0
            for i in range(0,self.m):
                tempb = tempb - self.Y[i]*self.X[i][j] + self.X[i][j]*math.pow(math.e,tempa[i])/(1+math.pow(math.e,tempa[i]))
            self.wd[j] = tempb/self.m
    def lossfunc(self):#求损失函数，优化的目的是尽可能最小化这个损失函数
        tempa = np.zeros((self.m, 1))
        for i in range(0, self.m):
            for j in range(0, self.n):
                tempa[i] = self.X[i][j] * self.w[j] + tempa[i]
        loss = 0
        for i in range(0,self.m):
            loss = loss + self.Y[i]*tempa[i] - math.log(1 + math.pow(math.e,tempa[i]))
        return loss
    def gradientdescent(self):#梯度优化的过程
        u = 1000
        lastloss = self.lossfunc()#当前系数对应的损失函数的值
        k = 0.15#步长
        count = 0
        while u > 0.005:
            print(k)
            self.differentialfunc()#对损失函数求导
            for i in range(0,self.n):
                self.w[i] = self.w[i] - k*self.wd[i]#按照梯度下降的方向改变系数值
            nowloss = self.lossfunc()
            u = math.fabs(nowloss - lastloss)
            #if nowloss - lastloss > 0.05:
             #   k = k/2
            lastloss = nowloss
            print(nowloss)
    def test(self,m,n,X,Y):#测试学习效果的函数
        self.m = m
        self.n = n
        self.X = X
        self.Y = Y
        plt.plot([0,-(self.w[2]/self.w[0])],[-(self.w[2]/self.w[1]),0])#逻辑回归的结果，用一条直线表示
        for i in range(0,self.m):
            if self.Y[i] == 0:
                plt.scatter(self.X[i][0], self.X[i][1], color='red')
            else:
                plt.scatter(self.X[i][0], self.X[i][1], color='blue')
        plt.show()
    def differentialfuncwithpunish(self,hyper):
        self.differentialfunc()
        for i in range(0,self.n):
            self.wd[i] = self.wd[i] + hyper*self.wd[i]
    def lossfunwithpunish(self,hyper):
        partloss = self.lossfunc()
        temp = 0
        for i in range(0,self.n):
            temp = temp + hyper*self.w[i]*self.w[i]/2
        loss = temp + partloss
        return loss
    def gradientdescentwithpunish(self,hyper):
        """
        带惩罚项的损失函数梯度下降过程
        :param hyper: 超参数，调节惩罚项的权重
        :return: 无
        """
        u = 1000
        lastloss = self.lossfunwithpunish(hyper)
        k = 1#步长
        while u > 0.001:
            self.differentialfuncwithpunish(hyper)
            for i in range(0,self.n):
                self.w[i] = self.w[i] - k*self.wd[i]
            nowloss = self.lossfunwithpunish(hyper)
            u = math.fabs(nowloss - lastloss)
            lastloss = nowloss
    def testwithpunish(self,m,n,X,Y):
        self.m = m
        self.n = n
        self.X = X
        self.Y = Y
        plt.plot([0, -(self.w[2] / self.w[0])], [-(self.w[2] / self.w[1]), 0])
        for i in range(0, self.m):
            if self.Y[i] == 0:
                plt.scatter(self.X[i][0], self.X[i][1], color='red')
            else:
                plt.scatter(self.X[i][0], self.X[i][1], color='blue')
        plt.show()










