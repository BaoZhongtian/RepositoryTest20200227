import os
import h5py
import numpy

if __name__ == '__main__':
    labelPath = r'D:\PythonProjects_Data\CMU_MOSEI\CMU_MOSEI_LabelsSentiment.csd'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/Step1_StartEndCut/'
    os.makedirs(savePath)

    labelData = h5py.File(name=labelPath, mode='r')
    # for sample in labelData['Sentiment Labels/data/zx4W0Vuus-I/intervals']:
    #     print(sample)
    for sampleName in labelData['Sentiment Labels/data'].items():
        sampleName = sampleName[0]
        labelTicket, timeTicket = [], []
        print(sampleName)

        with open(os.path.join(savePath, sampleName + '.csv'), 'w') as file:
            for sample in labelData['Sentiment Labels/data/%s/features' % sampleName]:
                labelTicket.append(sample)
            # labelTicket = numpy.reshape(numpy.argmax(labelTicket, axis=1), [-1, 1])
            for sample in labelData['Sentiment Labels/data/%s/intervals' % sampleName]:
                timeTicket.append(sample)
            totalTicket = numpy.concatenate([numpy.array(labelTicket), numpy.array(timeTicket)], axis=1)
            print(numpy.shape(totalTicket))
            for indexX in range(numpy.shape(totalTicket)[0]):
                for indexY in range(numpy.shape(totalTicket)[1]):
                    if indexY != 0: file.write(',')
                    file.write(str(totalTicket[indexX][indexY]))
                file.write('\n')

        # exit()
