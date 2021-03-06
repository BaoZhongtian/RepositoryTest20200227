import os
import numpy

if __name__ == '__main__':
    appointPart = 'Video'
    appointAttention = 'MonotonicAttention'
    if appointPart == 'Audio': comparePart = 'Video'
    if appointPart == 'Video': comparePart = 'Audio'

    attentionHotMapPath = 'D:/PythonProjects_Data/CMU_MOSEI_Result/Map-%s/BLSTM-W-%s-10/AttentionHotMap-%s-0099.npy'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI_Result/AttentionHotMap/%s/%s/' % (appointPart, appointAttention)
    if not os.path.exists(savePath): os.makedirs(savePath)

    rawTrainMap = numpy.load(attentionHotMapPath % (appointPart, appointAttention, 'Train'), allow_pickle=True)
    rawTestMap = numpy.load(attentionHotMapPath % (appointPart, appointAttention, 'Test'), allow_pickle=True)
    compareTrainMap = numpy.load(attentionHotMapPath % (comparePart, appointAttention, 'Train'), allow_pickle=True)
    compareTestMap = numpy.load(attentionHotMapPath % (comparePart, appointAttention, 'Test'), allow_pickle=True)

    trainAttentionHotMap, testAttentionHotMap = [], []
    for index in range(numpy.shape(rawTrainMap)[0]):
        extendLine = numpy.repeat(rawTrainMap[index], 1000)
        try:
            pointTicket = numpy.linspace(0, len(extendLine), num=len(compareTrainMap[index]) + 1)
            shrinkLine = [extendLine[int(v)] for v in pointTicket[:-1]]
        except:
            shrinkLine = [1]
        print('\rTreating Train %d' % index, end='')
        trainAttentionHotMap.append(shrinkLine)
    numpy.save(os.path.join(savePath, 'TrainAttentionHotMap.npy'), trainAttentionHotMap)

    for index in range(numpy.shape(rawTestMap)[0]):
        extendLine = numpy.repeat(rawTestMap[index], 1000)
        try:
            pointTicket = numpy.linspace(0, len(extendLine), num=len(compareTestMap[index]) + 1)
            shrinkLine = [extendLine[int(v)] for v in pointTicket[:-1]]
        except:
            shrinkLine = [1]
        print('\rTreating Test %d' % index, end='')
        testAttentionHotMap.append(shrinkLine)
    numpy.save(os.path.join(savePath, 'TestAttentionHotMap.npy'), testAttentionHotMap)
