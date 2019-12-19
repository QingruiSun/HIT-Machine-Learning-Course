import pandas as pd
import numpy as np

class data_cleaning:
    def __init__(self,data):
        self.data = data
    def drop_somecolumns(self):
        """
        除去一些没有序关系的值
        :return:
        """
        self.data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'income']
        self.data = self.data.drop(labels='workclass', axis=1)
        self.data = self.data.drop(labels='fnlwgt', axis=1)
        self.data = self.data.drop(labels='education', axis=1)
        self.data = self.data.drop(labels='occupation', axis=1)
        self.data = self.data.drop(labels='relationship', axis=1)
        self.data = self.data.drop(labels='native-country', axis=1)
        self.data = self.data.drop(labels='martial-status',axis=1)
    def funcsex(self,x):
        x = x.strip()
        if x == 'Female':
            return 0
        else:
            return 1
    def funcrace(self,x):
        x = x.strip()
        if x == 'White':
            return 0
        if x == 'Asian-Pac-Islander':
            return 1
        if x == 'Amer-Indian-Eskimo':
            return 2
        if x == 'Other':
            return 3
        if x == 'Black':
            return 4
    def funincome(self,x):
        x = x.strip()
        if x == '<=50K':
            return 0
        else:
            return 1
    def begin_cleaning(self):
        self.drop_somecolumns()
        self.data['sex'] = self.data['sex'].apply(lambda x:self.funcsex(x))
        self.data['race'] = self.data['race'].apply(lambda  x:self.funcrace(x))
        self.data['income'] = self.data['income'].apply(lambda x:self.funincome(x))
        x_point = np.array(self.data)
        x_point = np.delete(x_point,7,axis=1)#在原来的矩阵中删除最后一项收入项。
        y_value = np.array(self.data['income'])
        rownumber = x_point.shape[0]
        colnumber = x_point.shape[1]
        print('rownumber = ',rownumber)
        print('max = ',x_point.max(axis=0))
        x_point = np.array(x_point,dtype=float)
        count = 0
        #对一些取值过大的属性进行缩放，防止运算过程中产生溢出
        for i in range(0,rownumber):
            x_point[i][0] = float(x_point[i][0])/10
            x_point[i][1] = x_point[i][1]/10.0
            x_point[i][2] = x_point[i][2]
            x_point[i][3] = x_point[i][3]
            x_point[i][4] = x_point[i][4]/100000.0
            x_point[i][5] = x_point[i][5]/10000.0
            x_point[i][6] = x_point[i][6]/100.0
        B = np.ones(rownumber)
        x_point = np.insert(x_point,colnumber,values=B,axis=1)
        return x_point,y_value



