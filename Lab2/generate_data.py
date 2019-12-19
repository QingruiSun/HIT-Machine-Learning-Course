import numpy as np
import random
import matplotlib as pl


class data_genera:
    def generator(self,number,posnumber,mean_pos,mean_neg,cov11,cov12,cov21,cov22):
        """
        按照二元高斯分布来自动生成数据集
        :param number: 数据集样本数量
        :param posnumber: 样本中正例数量
        :param mean_pos: 正例样本的属性均值
        :param mean_neg: 反例样本的属性均值
        :param cov11: cov(正例样本集，正例样本集)
        :param cov12: cov（正例样本集,反例样本集)
        :param cov21: cov(反例样本集，正例样本集)
        :param cov22: cov(反例样本集，反例样本集)
        :return: x_point是样本集的属性值矩阵，y_value是生成样本的真实标记
        """
        x_point = []
        y_value = []
        for i in range(0,posnumber):
            tempxa,tempxb = np.random.multivariate_normal([mean_pos,mean_pos],[[cov11,cov12],[cov21,cov22]])
            x_point.append([tempxa,tempxb,1])
            y_value.append(1)
        for i in range(0,number-posnumber):
            tempxa,tempxb = np.random.multivariate_normal([mean_neg,mean_neg],[[cov11,cov12],[cov21,cov22]])
            x_point.append([tempxa,tempxb,1])
            y_value.append(0)
        for i in range(0,number):#将正例样本与反例样本随机混合
            j = random.randint(0,number-1)
            temppoint = x_point[i]
            x_point[i] = x_point[j]
            x_point[j] = temppoint
            tempvalue = y_value[i]
            y_value[i] = y_value[j]
            y_value[j] = tempvalue
        return x_point,y_value

