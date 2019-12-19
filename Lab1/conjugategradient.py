import numpy as np

class conjugateGradient(object):
    def __init__(self,X,T,hyper,delta=1e-6):
        self.X = X
        self.T = T
        self.hyper = hyper #惩罚参数
        self.delta = delta #迭代结果精度值
        #将问题转化成Ax = b这样解线性方程组的形式
        self.A = np.transpose(self.X) @ (self.X) +self.hyper*np.identity(len(self.T))
        self.b = np.transpose(self.X) @ (self.T)

    def solution(self):
        x = np.zeros(shape=(len(self.b),))
        r = self.b - self.A @ x
        p = r
        k = 0 #统计迭代次数
        while True:
            tempa = np.transpose(r) @ r
            tempb = np.transpose(p) @ self.A @ p
            a = tempa/tempb
            x = x + a * p
            rkk = r - a * self.A @ p
            loss = np.transpose(rkk) @ rkk
            print("loss= ",loss)
            if(loss < self.delta):
                break
            tempa = np.transpose(rkk) @ rkk
            tempb = np.transpose(r) @ r
            beta = tempa/tempb
            p = rkk + beta * p
            k = k +1
            r = rkk
        print(k)
        return x



