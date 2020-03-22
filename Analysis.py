import os
import numpy
import matplotlib.pylab as plt
from sklearn.metrics import f1_score
from Auxiliary.Tools import UA_Calculation, F1Score_Calculation

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI_Audio/Emotion1/BLSTM-W-MonotonicAttention-10-TestResult/'
    resultLine = []
    for fileName in os.listdir(loadPath):
        data = numpy.genfromtxt(fname=os.path.join(loadPath, fileName), dtype=float, delimiter=',')
        predict = data[:, 0:2]
        label = data[:, 2]

        confusionMatrix = numpy.zeros([2, 2])
        for index in range(numpy.shape(predict)[0]):
            confusionMatrix[1 - int(label[index])][1 - numpy.argmax(predict[index])] += 1
        f1Again = f1_score(y_true=label, y_pred=numpy.argmax(predict, axis=1))
        # print(f1Again)
        resultLine.append(
            [UA_Calculation(matrix=confusionMatrix), F1Score_Calculation(matrix=confusionMatrix), f1Again])
    plt.plot(resultLine)
    # plt.show()
    print(resultLine)
    print(confusionMatrix)
    print(numpy.max(resultLine, axis=0))
