import numpy as np
from generate_data import data_genera
from calculateneww import  calculate_neww

print('ds')
generatorins = data_genera()
datanumber = 100
"""
x_point,y_value = generatorins.generator(datanumber,50,2,1.2,0.4,0,0,0.3)
"""
x_point,y_value = generatorins.generator(datanumber,50,2,1.2,0.4,0,0,0.3)
x_point = np.array(x_point)
y_value = np.array(y_value)
calculator = calculate_neww(100,3,x_point,y_value)
"""
calculator.differentialfunc()
print(calculator.w)
calculator.gradientdescent()
print(calculator.w)
print(x_point.shape)
print(y_value.shape)
datanumber = 30
test_x_point,test_y_value = generatorins.generator(datanumber,15,2,1.2,0.4,0,0,0.3)
test_x_point = np.array(test_x_point)
test_y_value = np.array(test_y_value)
calculator.test(datanumber,3,test_x_point,test_y_value)
"""
datanumber = 30
test_x_point,test_y_value = generatorins.generator(datanumber,15,2,1.2,0.4,0,0,0.3)
"""
test_x_point,test_y_value = generatorins.generator(datanumber,15,2,1.2,0.4,0.4,0.4,0.3)
"""
test_x_point = np.array(test_x_point)
test_y_value = np.array(test_y_value)
calculator.gradientdescentwithpunish(0.05)
calculator.testwithpunish(datanumber,3,test_x_point,test_y_value)



