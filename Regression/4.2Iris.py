#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    # path = u'..\\算法讲师\\机器学习升级版\\数据\\4.2.iris.data'  # 数据文件路径
    path = './4.iris.data'  # 数据文件路径

    '''
    1. Read data manually
    '''
    # f = file(path)
    # x = []
    # y = []
    # for d in f:
    #     d = d.strip() # string (remove space \n)
    #     if d:
    #         d =  d.split(',') # list
    #         # print d
    #         y.append(d[-1])
    #         x.append(map(float, d[:-1])) # convert data from string to float
    # # print 'x = \n', x
    # # print 'y = \n', y

    '''
    Clean data manually
    Convert x and y from list to numpy array
    Convert string y to int y
    '''
    # x = np.array(x)
    # # print 'numpy x = ', x
    # y = np.array(y)
    # print 'numpy y = ', y
    # y[y == 'Iris-setosa'] = 0
    # y[y == 'Iris-versicolor'] = 1
    # y[y == 'Iris-virginica'] = 2
    # print 'numpy y = ', y
    # y = y.astype(dtype = np.int) 
    # print 'numpy y = ', y
   
    
    '''
    # 使用sklean的数据预处理
    2. Process raw data by sklean
    '''
    # df = pd.read_csv(path)
    # x = df.values[:, :-1]
    # y = df.values[:, -1]
    # print 'x = \n', x
    # print 'y = \n', y
    # le = preprocessing.LabelEncoder()
    # le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    # print le.classes_
    # y = le.transform(y)
    # print y, type(y)


    '''
    3. Process data by numpy
    '''
    # 路径，浮点型数据，逗号分隔，第4列使用函数iris_type单独处理
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})

    # 将数据的0到3列组成x，第4列得到y
    # 4 means split in the 4th element, where first 4 elements is x, the other is y
    x, y = np.split(data, (4,), axis=1) # axis = 0 is for row split, 1 is for column split

    # 为了可视化，仅使用前两列特征
    x = x[:, :2]

    # print x
    # print y

    '''
    Train Model
    '''
    # x = StandardScaler().fit_transform(x) # 数据标准化
    logreg = LogisticRegression()   # Logistic回归模型
    logreg.fit(x, y.ravel())        # 根据数据[x,y]，计算回归参数

    # 等价形式
    # logreg = Pipeline( [ ('sc', StandardScaler()), ('clf', LogisticRegression())  ] )
    # logreg.fit(x, y.ravel())        # 根据数据[x,y]，计算回归参数


    '''
    Plot figures
    '''
    # 画图
    N, M = 500, 500     # 横纵各采样多少个值
    # N, M = 20, 20     # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()   # 第0列的范围 [:, 0] means the 0th column
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()   # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)   # 测试点
    # print x1
    # print x2
    # print x_test

    # # 无意义，只是为了凑另外两个维度
    # x3 = np.ones(x1.size) * np.average(x[:, 2])
    # x4 = np.ones(x1.size) * np.average(x[:, 3])
    # x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)  # 测试点

    y_hat = logreg.predict(x_test)                  # 预测值
    y_hat = y_hat.reshape(x1.shape)                 # 使之与输入的形状相同
    # print y_hat
    plt.pcolormesh(x1, x2, y_hat, cmap=plt.cm.Spectral, alpha=0.5)  # 预测值的显示Paired/Spectral/coolwarm/summer/spring/OrRd/Oranges
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=plt.cm.prism)  # 样本的显示
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.show()

    '''
    Show Summary
    '''
    # 训练集上的预测结果
    # y_hat = dt_clf.predict(x)
    y_hat = logreg.predict(x)
    y = y.reshape(-1)       # 此转置仅仅为了print时能够集中显示
    # print y_hat.shape       # 不妨显示下y_hat的形状
    # print y.shape
    result = (y_hat == y)   # True则预测正确，False则预测错误
    print y_hat
    print y
    print result
    c = np.count_nonzero(result)    # 统计预测正确的个数
    print c
    print 'Accuracy: %.2f%%' % (100 * float(c) / float(len(result)))


    '''
    explain ravel() and reshape() are equal
    '''
    z = np.arrange(10)
    z.shape = -1, 1
    print z
    print 'ravel = \n', y.ravel()
    print 'reshape = \n', y.reshape(1, -1)
