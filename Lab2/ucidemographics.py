import pandas as pd
from datacleaning import data_cleaning
import numpy as np
from calculateneww import calculate_neww

data = pd.read_csv('adult.data')
data_cleaning_ins = data_cleaning(data)
x_point,y_value = data_cleaning_ins.begin_cleaning()
dimension = 8
trainnumber = 1000
x_point_train = np.array(x_point[0:trainnumber,0:dimension])
y_value_train = np.array(y_value[0:trainnumber])
calculator = calculate_neww(trainnumber,dimension,x_point_train,y_value_train)
calculator.gradientdescentwithpunish(0.05)
w = calculator.w#获得逻辑回归的系数矩阵
print(w)
x_point_test = np.array(x_point[trainnumber:2*trainnumber,0:dimension])
y_value_test = np.array(y_value[trainnumber:2*trainnumber])
accuratenumber = 0
for i in range(0,trainnumber):
    loss = 0
    for j in range(0,dimension):
        loss = loss + x_point_test[i][j]*w[j]+0.025*w[j]*w[j]
    if (loss>=0 and (y_value_test[i] == 1)) or (loss<0 and (y_value_test[i] == 0)):
        accuratenumber = accuratenumber + 1
print('accuratenumber = ',accuratenumber)


