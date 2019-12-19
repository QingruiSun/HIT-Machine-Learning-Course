import numpy as np
import math
import random
import matplotlib.pyplot as plt
import analytical
import gradientdescent
import conjugategradient

def data_generator(n,startvalue):
    """
    产生数据集
    :param n: 数据集中数据点个数
    :param startvalue: 数据开始点
    :return: 数据集中每个数据点的坐标
    """
    x = np.linspace(startvalue,startvalue+1,n)
    t = np.sin(2*np.pi*x)+np.random.normal(loc=0,scale=0.2,size=n)
    return x,t

def transform(X,degree):
    """
    将行向量X转化成便于计算的范德蒙德矩阵
    :param X: 行向量
    :param degree: 阶数
    :return: 转化后的范德蒙德矩阵
    """
    X_T = X.transpose()
    X = np.transpose([X])
    features = [np.ones(len(X))]
    for i in range(0, degree):
        features.append(np.multiply(X_T, features[i]))
    return np.asarray(features).transpose()
def fit_func(X_Test,w):
    """
    用来获取测试数据在拟合函数上的表现
    :param X_Test:  a m+1*m+1 matrix
    :param w: 获得拟合函数的系数
    :return:  测试数据在拟合函数上的值
    """
    return np.dot(X_Test,w)

def get_rms(Y_Test,T):
    """
    求得rms误差
    :param Y_Test: 测试数据的表现
    :param T: 数据点真实值
    :return: 测试数据的rms误差
    """
    square_item = np.square(Y_Test-T)
    square_sum = np.sum(square_item)
    return np.sqrt(square_sum/len(T))

def no_punishment():
    """
    没有惩罚项的拟合
    """
    train_number = 10
    test_number = 100
    degree = 9
    X_training,T_training = data_generator(train_number,0)
    X_test = np.linspace(0,1,test_number)
    X_Train = transform(X_training,degree)
    X_Test = transform(X_test,degree)
    Y = np.sin(2*np.pi*X_test)
    # X_transpose = np.transpose(X_Train)
    # temp = np.dot(X_transpose,X_Train)
    #temp_inver = np.linalg.inv(temp)
    # W = temp_inver.dot(temp).dot(X_transpose).dot(T_training)
    anaSolution = analytical.AnalyticalSolution(X_Train,T_training)
    W = anaSolution.fitting()
    plt.figure(figsize=(20,10))
    plt.scatter(X_training,T_training,label="training set")
    plt.plot(X_test,Y,label="sin2*pi*x")
    Y_Test = fit_func(X_Test,W)
    plt.plot(X_test,Y_Test,label="test result")
    plt.legend()
    plt.show()

def get_para():
    """
    用来获取惩罚参数的最佳值
    """
    train_number = 10
    degree = 1
    X_training,T_Train = data_generator(train_number,0)
    X_test = np.linspace(0,1,100)
    Y = np.sin(2*np.pi*X_test)
    X_Test = transform(X_test,degree)
    X_Train = transform(X_training,degree)
    anaSolution = analytical.AnalyticalSolution(X_Train, T_Train)
    hyperTestList = []
    hyperTrainList = []
    hyperList = range(-50, 1)
    for hyper in hyperList:
        w_analytical_with_regulation = anaSolution.fitting_with_regulation(
            np.exp(hyper))
        T_test = fit_func(X_Test, w_analytical_with_regulation)
        hyperTestList.append(get_rms(T_test, Y))
        bestHyper = hyperList[np.where(hyperTestList ==
                                   np.min(hyperTestList))[0][0]]
    print("bestHyper:", bestHyper, np.min(hyperTestList))
    return bestHyper

def have_punishment():
    """
    有惩罚项的拟合
    """
    besthyper = -7
    train_number = 10
    degree = 18
    X_training,T_training = data_generator(train_number,0)
    X_Train = transform(X_training,degree)
    X_test = np.linspace(0,1,100)
    X_Test = transform(X_test,degree)
    Y_Test = np.sin(2*np.pi*X_test)
    anasolution = analytical.AnalyticalSolution(X_Train,T_training)
    w_ana = anasolution.fitting_with_regulation(np.exp(besthyper))
    plt.scatter(X_training,T_training)
    plt.plot(X_test,Y_Test,label="sin(2*pi*x)")
    plt.plot(X_test,fit_func(X_Test,w_ana),label="test line")
    plt.legend()
    plt.show();

def gradient_desc():
    """
    梯度下降法的拟合
    """
    train_number = 7
    test_number = 100
    degree = 9
    besthyper = -7
    X_training,T_training = data_generator(train_number,0)
    X_Train = transform(X_training,degree)
    X_test = np.linspace(0,1,100)
    X_Test = transform(X_test,degree)
    Y = np.sin(2*np.pi*X_test)
    gd = gradientdescent.GradientDescent(X_Train,T_training,np.exp(besthyper))
    W = gd.fitting(np.zeros(degree+1))
    plt.scatter(X_training,T_training,label="training data")
    plt.plot(X_test,fit_func(X_Test,W),label="gradient descent")
    plt.plot(X_test,Y,label="sin*2*pi")
    plt.legend()
    plt.show()

def conjugate_gra():
    """
    共轭梯度下降法的拟合
    """
    train_number = 10
    test_number = 100
    degree = 9
    hyper = get_para()
    X_training,T_training = data_generator(train_number,0)
    X_Train = transform(X_training,degree)
    X_test = np.linspace(0,1,100)
    X_Test = transform(X_test,degree)
    Y = np.sin(2*np.pi*X_test)
    cg = conjugategradient.conjugateGradient(X_Train,T_training,np.exp(hyper))
    w = cg.solution()
    Y_Test = fit_func(X_Test,w)
    Y_Train = fit_func(X_Train,w)
    lossvector = Y_Train - T_training
    loss = np.transpose(lossvector) @ lossvector
    print(loss)
    plt.scatter(X_training,T_training,label="training set")
    plt.plot(X_test,Y,label="sin(2*pi*x)")
    plt.plot(X_test,Y_Test,label="testing set")
    plt.legend()
    plt.show()


x,t = data_generator(10,0)
X = transform(x,2)
get_para()
have_punishment()
#gradient_desc()
#no_punishment()

