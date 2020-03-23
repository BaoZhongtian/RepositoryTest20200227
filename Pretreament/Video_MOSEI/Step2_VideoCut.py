import os
import h5py
import numpy

if __name__ == '__main__':
    labelPath = 'D:/PythonProjects_Data/CMU_MOSEI/Step1_StartEndCut/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/Step2_VideoCut/'
    if not os.path.exists(savePath): os.makedirs(savePath)
    originData = h5py.File(name=r'D:\PythonProjects_Data\CMU_MOSEI\CMU_MOSEI_VisualFacet42.csd', mode='r')

    # for sample in originData['FACET 4.2/data/zx4W0Vuus-I']:
    #     print(sample)

    for fileName in os.listdir(labelPath)[3::4]:
        labelData = numpy.reshape(numpy.genfromtxt(fname=os.path.join(labelPath, fileName), delimiter=',', dtype=float),
                                  [-1, 3])
        print(fileName, numpy.shape(labelData))

        currentData = originData['FACET 4.2/data/%s/features' % fileName.replace('.csv', '')]
        currentInterval = originData['FACET 4.2/data/%s/intervals' % fileName.replace('.csv', '')]
        print(numpy.shape(currentData), numpy.shape(currentInterval))

        for sampleCounter in range(numpy.shape(labelData)[0]):
            sampleData = []
            for index in range(numpy.shape(currentInterval)[0]):
                if labelData[sampleCounter][-2] <= currentInterval[index][0] and \
                        currentInterval[index][1] <= labelData[sampleCounter][-1]:
                    sampleData.append(currentData[index])
            print(labelData[sampleCounter][-2], labelData[sampleCounter][-1], numpy.shape(sampleData))

            with open(os.path.join(savePath, fileName.replace('.csv', '%02d.csv' % sampleCounter)), 'w') as file:
                for indexX in range(numpy.shape(sampleData)[0]):
                    for indexY in range(numpy.shape(sampleData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(sampleData[indexX][indexY]))
                    file.write('\n')
