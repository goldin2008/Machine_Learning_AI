#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    # path = u'..\\算法讲师\\机器学习升级版\\数据\\4.1Advertising.csv'
    # path = '/Users/yuleinku/Dropbox/15th/code/4.Advertising.csv'
    path = './4.Advertising.csv'

    '''
    Raw data process
    Use 3 diff ways to read csv data
    # Python自带库
    '''
    # f = file(path, 'rb')
    # print f
    # d = csv.reader(f)
    # for line in d:
    #     print line
    # f.close()

    '''
    # numpy读入
    '''
    # p = np.loadtxt(path, delimiter=',', skiprows=1)
    # print p

    '''
    # pandas读入
    '''
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    # print 'x = ' , x
    # print 'y = ', y

    '''
    # 绘制1
    plot 3 figures in 1 figure
    '''
    # plt.plot(data['TV'], y, 'ro', label='TV')
    # plt.plot(data['Radio'], y, 'g^', label='Radio')
    # plt.plot(data['Newspaper'], y, 'b*', label='Newspaer')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()

    '''
    # 绘制2
    plot 3 figures in 3 diff figures
    '''
    # plt.figure(figsize=(9,12))
    # plt.subplot(311)
    # plt.plot(data['TV'], y, 'ro')
    # plt.title('TV')
    # plt.grid()
    # plt.subplot(312)
    # plt.plot(data['Radio'], y, 'g^')
    # plt.title('Radio')
    # plt.grid()
    # plt.subplot(313)
    # plt.plot(data['Newspaper'], y, 'b*')
    # plt.title('Newspaper')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()


    '''
    # Train and Test Regression Model
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # print x_train, y_train
    '''
    print traiing and testing data information (n, m) where n is # of data, and m is # of dim of data
    '''
    print x_train.shape
    print x_test.shape

    '''
    Train model
    '''
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print model
    print linreg.coef_
    print linreg.intercept_

    '''
    Test model
    http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html
    http://scikit-learn.org/stable/modules/cross_validation.html
    '''
    # y_hat = linreg.predict(x_test)
    y_hat = linreg.predict( np.array(x_test) )
    mse = np.average((y_hat - y_test) ** 2)    # Mean Squared Error
    rmse = np.sqrt(mse)     # Root Mean Squared Error
    print mse, rmse

    '''
    plot and compare real data and predicted data
    '''
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
