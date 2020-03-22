import os
import numpy

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/Step1_StartEndCut/'

    labelCounter = {1: 0, 0: 0}
    for fileName in os.listdir(loadPath):
        data = numpy.reshape(numpy.genfromtxt(fname=os.path.join(loadPath, fileName), dtype=float, delimiter=','),
                             [-1, 3])
        for sample in data:
            # print(sample)
            if sample[0] > 0:
                labelCounter[1] += 1
            else:
                labelCounter[0] += 1
    for sample in labelCounter.keys():
        print(sample, labelCounter[sample])
