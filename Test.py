import numpy
from Auxiliary.Loader import Loader_Total

if __name__ == '__main__':
    trainAttentionHotMap = numpy.load(
        r'D:\PythonProjects_Data\AttentionHotMap\Video\StandardAttention\TrainAttentionHotMap.npy',
        allow_pickle=True)
    trainDataset, frozenDataset, testDataset = Loader_Total(appointPart='Audio', batchSize=1)
    counter = 0
    for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(frozenDataset):
        print(batchNumber, numpy.shape(batchData), batchSeq, numpy.shape(trainAttentionHotMap[counter]))
        counter += 1
        if counter == 10: break
