import os
import numpy

if __name__ == '__main__':
    dataPath = 'D:/PythonProjects_Data/CMU_MOSEI/TotalPart/Video/'
    labelPath = 'D:/PythonProjects_Data/CMU_MOSEI/Step1_StartEndCut/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/Transform_Data_Video/'
    if not os.path.exists(savePath): os.makedirs(savePath)

    for part in os.listdir(dataPath):
        totalData, totalLabel = [], []
        for fileName in os.listdir(os.path.join(dataPath, part)):
            print(part, fileName)
            currentData = numpy.reshape(
                numpy.genfromtxt(fname=os.path.join(dataPath, part, fileName), dtype=float, delimiter=','), [-1, 35])
            currentLabel = numpy.reshape(
                numpy.genfromtxt(fname=os.path.join(labelPath, fileName[:-7] + '.csv'), dtype=float,
                                 delimiter=','), [-1, 3])

            counter = int(fileName[-6:-4])
            totalData.append(currentData)
            if currentLabel[counter][0] > 0:
                totalLabel.append(1)
            else:
                totalLabel.append(0)
            # print(numpy.shape(totalData), numpy.shape(totalLabel))
            # exit()
        print(numpy.shape(totalData), numpy.shape(totalLabel), numpy.sum(totalLabel, axis=0))
        numpy.save(file=os.path.join(savePath, '%s-Data.npy' % part), arr=totalData)
        numpy.save(file=os.path.join(savePath, '%s-Label.npy' % part), arr=totalLabel)
