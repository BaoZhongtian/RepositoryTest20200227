import os
import numpy
import matplotlib.pylab as plt


def MAE_Calculate(predict, label):
    return numpy.average(numpy.abs(numpy.subtract(predict, label)))


def RMSE_Calculate(predict, label):
    return numpy.sqrt(numpy.average(numpy.square(numpy.subtract(predict, label))))


if __name__ == '__main__':
    dataPath = 'D:/PythonProjects_Data/CMU_MOSEI_Audio/BLSTM-W-MonotonicAttention-10-TestResult/'
    maeLine, rmseLine, f1Line, uaLine = [], [], [], []

    for fileName in os.listdir(dataPath):
        print(fileName)
        data = numpy.genfromtxt(fname=os.path.join(dataPath, fileName), dtype=float, delimiter=',')
        predict, label = data[:, 0:6] / 10, data[:, 6:] / 10
        maeLine.append(MAE_Calculate(predict=predict, label=label))
        rmseLine.append(RMSE_Calculate(predict=predict, label=label))

        confusionMatrix = numpy.zeros([6, 2, 2])

        for indexX in range(numpy.shape(predict)[0]):
            for indexY in range(numpy.shape(predict)[1]):
                if predict[indexX][indexY] <= 0:
                    row = 1
                else:
                    row = 0
                if label[indexX][indexY] <= 0:
                    line = 1
                else:
                    line = 0
                confusionMatrix[indexY][line][row] += 1

        # for indexX in range(6):
        #     print(confusionMatrix[indexX])
        # exit()

        currentF1, currentUA = [], []
        for indexX in range(6):
            currentUA.append((confusionMatrix[indexX][0][0] / numpy.sum(confusionMatrix[indexX][0]) +
                              confusionMatrix[indexX][1][1] / numpy.sum(confusionMatrix[indexX][1])) / 2)

            precision = confusionMatrix[indexX][0][0] / numpy.sum(confusionMatrix[indexX][0])
            recall = confusionMatrix[indexX][0][0] / (confusionMatrix[indexX][0][0] + confusionMatrix[indexX][1][0])
            currentF1.append(2 * (precision * recall) / (precision + recall))
        uaLine.append(currentUA)
        f1Line.append(currentF1)
    # plt.plot(maeLine)
    # plt.plot(rmseLine)
    print(numpy.min(maeLine), '\t', numpy.min(rmseLine), end='\t')
    for index in range(6):
        print(numpy.max(uaLine, axis=0)[index], '\t', numpy.max(f1Line, axis=0)[index], end='\t')
