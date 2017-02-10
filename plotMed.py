# -*- coding:utf-8 -*-
__author__ = 'mengmeng'

import matplotlib.pyplot as plt
import numpy as np


def plot(inputFilePath, tag):
    f = open(inputFilePath, 'r')
    file = f.readlines()
    xTestLen = int(file[0:1][0])
    predict_resutl = []
    for line in file[1:2]:
        ssr = line.split()
        for i in np.arange(len(ssr)):
            # if float(ssr[i]) < 0.5:
            #     predict_resutl.append(0)
            # elif float(ssr[i]) >= 0.5:
            #     predict_resutl.append(1)
            predict_resutl.append(float(ssr[i]))
    yTest = []
    actual_result = []
    for line in file[2:]:
        ssr = line.split()
        for i in np.arange(len(ssr)):
            y = []
            y.append(ssr[i])
            yTest.append(y)
            actual_result.append(float(ssr[i]))

    # num = 0
    # for j in np.arange(len(actual_result)):
    #     if actual_result[j] - predict_resutl[j] == 0.0:
    #         num = num + 1
    # print num
    # print len(actual_result)
    # accuracy = float(num) / len(actual_result)
    # print "##########" + str(tag) + "accuracy###############: " + str(accuracy)

    f.close()
    plt.figure()
    plt.plot(np.arange(0, xTestLen), predict_resutl, 'ro--', label='predict result')
    plt.plot(np.arange(0, xTestLen), yTest, 'ko-', label='real result')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(tag)
    plt.savefig(tag + '.png')
    plt.show()
    plt.close()


filePath = "/Users/mengmeng"
tag = "mcihcmatrix"
inputFilePath = filePath + "/" + tag + ".txt"
plot(inputFilePath, tag)
