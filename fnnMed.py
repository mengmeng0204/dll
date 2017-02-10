# coding=utf-8
__author__ = 'mengmeng'

import multiprocessing
from time import ctime

from PIL import Image
from pylab import *
import numpy as np

from pybrain.structure import *
from pybrain.datasets import *
from pybrain.supervised.trainers import BackpropTrainer


# def generate_data():
#     """generate original data of u and y"""
#     u = np.random.uniform(-1, 1, 200)
#     y = []
#     former_y_value = 0
#     for i in np.arange(0, 200):
#         y.append(former_y_value)
#         next_y_value = (29 / 40) * np.sin(
#             (16 * u[i] + 8 * former_y_value) / (3 + 4 * (u[i] ** 2) + 4 * (former_y_value ** 2))) \
#                        + (2 / 10) * u[i] + (2 / 10) * former_y_value
#         former_y_value = next_y_value
#     return u, y


def generate_pic(filepath):
    f = open(filepath, 'r')
    u = []
    y = []
    while True:
        line = f.readline()
        if line:
            im = array(Image.open(line.split(" ")[0]).convert('L'), 'f')
            im.resize((256, 168))
            # 变成二维矩阵
            mtr = np.array(im)
            u.append(mtr)
            y.append(line.split(" ")[1])
        else:
            break
    f.close()
    return u, y


def network():
    # create a neural network
    fnn = FeedForwardNetwork()

    # create three layers, input layer:2 input unit; hidden layer: 10 units; output layer: 1 output
    inLayer = LinearLayer(256 * 168, name='inLayer')
    hiddenLayer0 = SigmoidLayer(10, name='hiddenLayer0')
    outLayer = LinearLayer(1, name='outLayer')

    # add three layers to the neural network
    fnn.addInputModule(inLayer)
    fnn.addModule(hiddenLayer0)
    fnn.addOutputModule(outLayer)

    # link three layers
    in_to_hidden0 = FullConnection(inLayer, hiddenLayer0)
    hidden0_to_out = FullConnection(hiddenLayer0, outLayer)

    # add the links to neural network
    fnn.addConnection(in_to_hidden0)
    fnn.addConnection(hidden0_to_out)

    # make neural network come into effect
    fnn.sortModules()
    return fnn


def train(tag, inputFilePath, outputFilePath):
    resultFilePath = outputFilePath + '/' + tag + 'result.txt'
    matrixFilePath = outputFilePath + '/' + tag + 'matrix.txt'
    file_matrix = open(matrixFilePath, 'w+')

    # obtain the original data
    u, y = generate_pic(inputFilePath)

    fnn = network()
    # definite the dataset as two input , one output
    # DS = SupervisedDataSet(256 * 168, 1)
    DS = ClassificationDataSet(256 * 168, 1)

    # add data element to the dataset
    for i in np.arange(len(u)):
        m = []
        try:
            for j in np.arange(256):
                for k in np.arange(168):
                    m.append(u[i][j][k])
        except Exception, e:
            print " error number: ", i
            continue
        DS.addSample(m, y[i])

    # you can get your input/output this way
    X = DS['input']
    Y = DS['target']

    # split the dataset into train dataset and test dataset
    dataTrain, dataTest = DS.splitWithProportion(0.9)
    xTrain, yTrain = dataTrain['input'], dataTrain['target']
    xTest, yTest = dataTest['input'], dataTest['target']

    # train the NN
    # we use BP Algorithm
    # verbose = True means print th total error
    trainer = BackpropTrainer(fnn, dataTrain, verbose=True, learningrate=0.01)

    # set the epoch times to make the NN  fit
    trainerrors, validerrors = trainer.trainUntilConvergence(maxEpochs=20)
    file_object = open(resultFilePath, 'w+')
    try:
        for i in np.arange(len(trainerrors)):
            file_object.write('train-errors:' + str(trainerrors[i]) + '\n')
            file_object.write('valid-errors:' + str(validerrors[i]) + '\n')
    except Exception, e:
        print e.message
    predict_resutl = []

    for i in np.arange(len(xTest)):
        prediction = fnn.activate(xTest[i])
        predict_resutl.append(prediction[0])
        print "the prediction number is: " + str(prediction[0]) + " ,the real number is:  " + str(yTest[i][0])
        file_object.write(
            "the prediction number is :" + str(prediction[0]) + " the real number is:  " + str(yTest[i][0]) + "\n")
        # print(predict_resutl)
    file_object.close()

    file_matrix.write(str(len(xTest)) + '\n')
    for pre in predict_resutl:
        file_matrix.write(str(pre) + ' ')
    file_matrix.write('\n')
    for y in yTest:
        file_matrix.write(str(y[0]) + ' ')
    file_matrix.close()

    return xTest, predict_resutl, yTest


def param(fnn):
    for mod in fnn.modules:
        print ("Module:", mod.name)
        if mod.paramdim > 0:
            print ("--parameters:", mod.params)
        for conn in fnn.connections[mod]:
            print ("-connection to", conn.outmod.name)
            if conn.paramdim > 0:
                print ("- parameters", conn.params)
        if hasattr(fnn, "recurrentConns"):
            print ("Recurrent connections")
            for conn in fnn.recurrentConns:
                print ("-", conn.inmod.name, " to", conn.outmod.name)
                if conn.paramdim > 0:
                    print ("- parameters", conn.params)


inputfilepath_data = "/Users/mengmeng/data1.txt"
inputfilepath_adhc = "/Users/mengmeng/adhc.txt"
inputfilepath_admci = "/Users/mengmeng/admci.txt"
inputfilepath_mcihc = "/Users/mengmeng/mcihc.txt"
outputFilePath = "/Users/mengmeng"
# plot('adhc', inputfilepath, outputfilepath)

print "start %s" % ctime()
# train("adhc", inputfilepath_adhc, outputFilePath)
train("admci", inputfilepath_admci, outputFilePath)
train("mcihc", inputfilepath_mcihc, outputFilePath)
print "all over %s" % ctime()

# processList = []
# p1 = multiprocessing.Process(target=train, args=('adhc', inputfilepath_adhc, outputFilePath,))
# processList.append(p1)
# p2 = multiprocessing.Process(target=train, args=('admci', inputfilepath_admci, outputFilePath,))
# processList.append(p2)
# p3 = multiprocessing.Process(target=train, args=('mcihc', inputfilepath_mcihc, outputFilePath,))
# processList.append(p3)
#
# if __name__ == '__main__':
#     print "start %s" % ctime()
#     for p in processList:
#         p.start()
#     for p in processList:
#         p.join()
#
#     print "all over %s" % ctime()
