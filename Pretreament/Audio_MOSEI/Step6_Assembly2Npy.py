import os
import numpy

if __name__ == '__main__':
    dataPath = 'D:/PythonProjects_Data/CMU_MOSEI/Step5_Normalization/'
    labelPath = 'D:/PythonProjects_Data/CMU_MOSEI/Step1_StartEndCut_Emotion/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/Data_Audio/'
    if not os.path.exists(savePath): os.makedirs(savePath)

    for part in os.listdir(dataPath):
        totalData, totalLabel = [], []
        for fileName in os.listdir(os.path.join(dataPath, part)):
            print(part, fileName)
            currentData = numpy.genfromtxt(fname=os.path.join(dataPath, part, fileName), dtype=float, delimiter=',')
            currentLabel = numpy.reshape(
                numpy.genfromtxt(fname=os.path.join(labelPath, fileName[:-7] + '.csv'), dtype=float,
                                 delimiter=','), [-1, 8])

            counter = int(fileName[-6:-4])
            totalData.append(currentData)
            totalLabel.append(currentLabel[counter][0:6])
            # print(numpy.shape(totalData), numpy.shape(totalLabel))
            # exit()
        print(numpy.shape(totalData), numpy.shape(totalLabel))
        numpy.save(file=os.path.join(savePath, '%s-Data.npy' % part), arr=totalData)
        numpy.save(file=os.path.join(savePath, '%s-Label.npy' % part), arr=totalLabel)
