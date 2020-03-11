import os
import numpy

if __name__ == '__main__':
    labelPath = 'D:/PythonProjects_Data/CMU_MOSEI/Metadata/'
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/Step1_StartEndCut_Emotion/'

    for part in ['train', 'valid', 'test']:
        dictionary = set()
        for fileName in os.listdir(os.path.join(labelPath, 'standard_%s_fold' % part)):
            dictionary.add(fileName.replace('.json', '.csv'))

        labelNumber = numpy.zeros(6)
        compareList = numpy.zeros(6)
        for fileName in os.listdir(loadPath):
            # if fileName not in dictionary: continue
            data = numpy.reshape(numpy.genfromtxt(fname=os.path.join(loadPath, fileName), dtype=float, delimiter=','),
                                 [-1, 8])
            data = data[:, 0:6]
            for indexX in range(numpy.shape(data)[0]):
                for indexY in range(numpy.shape(data)[1]):
                    if data[indexX][indexY] > compareList[indexY]:
                        labelNumber[0] += 1
                        break
            # print(fileName)
        # print(labelNumber)
        for sample in labelNumber:
            print(sample, end='\t')
        print()
