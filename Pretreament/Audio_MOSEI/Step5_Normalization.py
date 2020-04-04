import os
import numpy
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/CMU_MOSEI/AudioPart/Step4EX_SpectrumGeneration/'
    savepath = 'D:/PythonProjects_Data/CMU_MOSEI/AudioPart/Step5EX_Normalization/'

    totalData = []

    for partName in os.listdir(loadpath):
        for filename in os.listdir(os.path.join(loadpath, partName)):
            data = numpy.genfromtxt(fname=os.path.join(loadpath, partName, filename),
                                    dtype=float, delimiter=',')
            totalData.extend(data)
        print('Loading', partName, numpy.shape(totalData))

    print(numpy.shape(totalData))
    totalData = scale(totalData)
    startPosition = 0

    for partName in os.listdir(loadpath):
        os.makedirs(os.path.join(savepath, partName))
        for filename in os.listdir(os.path.join(loadpath, partName)):
            data = numpy.genfromtxt(fname=os.path.join(loadpath, partName, filename),
                                    dtype=float, delimiter=',')
            writeData = totalData[startPosition:startPosition + len(data)]

            with open(os.path.join(savepath, partName, filename), 'w') as file:
                for indexX in range(numpy.shape(writeData)[0]):
                    for indexY in range(numpy.shape(writeData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(writeData[indexX][indexY]))
                    file.write('\n')

            startPosition += len(data)
        print('Writing', partName, startPosition)
