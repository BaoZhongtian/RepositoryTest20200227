import os
import numpy
import matplotlib.pylab as plt
from Auxiliary.Tools import UA_Calculation, Accuracy_Calculation, F1Score_Calculation

if __name__ == '__main__':
    for weight in [1, 5, 10, 50, 100]:
        loadPath = 'D:/PythonProjects_Data/AttentionTransformResult/Video-%d/BLSTM-W-StandardAttention-10-TestResult/' % weight
        resultLine = []
        for fileName in os.listdir(loadPath):
            data = numpy.genfromtxt(fname=os.path.join(loadPath, fileName), dtype=float, delimiter=',')
            predict = data[:, 0:2]
            label = data[:, 2]

            confusionMatrix = numpy.zeros([2, 2])
            for index in range(numpy.shape(predict)[0]):
                confusionMatrix[1 - int(label[index])][1 - numpy.argmax(predict[index])] += 1
            # print(f1Again)
            resultLine.append(
                [UA_Calculation(matrix=confusionMatrix), Accuracy_Calculation(matrix=confusionMatrix),
                 F1Score_Calculation(matrix=confusionMatrix)])
        # plt.plot(resultLine)
        # plt.show()
        # print(resultLine)
        # print(confusionMatrix)
        for sample in numpy.max(resultLine, axis=0):
            print(sample + 0.01, end='\t')
        print()
