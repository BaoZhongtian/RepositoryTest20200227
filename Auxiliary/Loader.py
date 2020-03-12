import os
import torch
import numpy
from torch.utils import data as torch_utils_data


class Collate_FluctuateLen:
    def __init__(self, labelPadding=False):
        self.labelPadding = labelPadding

    def __DataTensor(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros([maxLen - len(dataInput), numpy.shape(dataInput)[1]],
                                                         dtype=torch.float)], axis=0)

    def __LabelTensor(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros([maxLen - len(dataInput)], dtype=torch.float)], axis=0)

    def __collateOnlyData(self, batch):
        xs = [v[0] for v in batch]
        ys = torch.FloatTensor([v[1] for v in batch])
        seq_lengths = torch.LongTensor([v for v in map(len, xs)])
        max_len = max([len(v) for v in xs])
        xs = numpy.array([self.__DataTensor(dataInput=v, maxLen=max_len) for v in xs], dtype=float)
        xs = torch.from_numpy(xs)
        return xs, seq_lengths, ys

    def __call__(self, batch):
        return self.__collateOnlyData(batch=batch)


class Dataset_General(torch_utils_data.Dataset):
    def __init__(self, data, label):
        self.data, self.label = data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def Loader_Audio(batchSize=32, multiply=10):
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/Data_Audio/'
    trainData = numpy.load(file=os.path.join(loadPath, 'Train-Data.npy'), allow_pickle=True).tolist()
    trainLabel = (numpy.load(file=os.path.join(loadPath, 'Train-Label.npy'), allow_pickle=True) * multiply).tolist()
    validData = numpy.load(file=os.path.join(loadPath, 'Valid-Data.npy'), allow_pickle=True).tolist()
    validLabel = (numpy.load(file=os.path.join(loadPath, 'Valid-Label.npy'), allow_pickle=True) * multiply).tolist()
    testData = numpy.load(file=os.path.join(loadPath, 'Test-Data.npy'), allow_pickle=True).tolist()
    testLabel = (numpy.load(file=os.path.join(loadPath, 'Test-Label.npy'), allow_pickle=True) * multiply).tolist()
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(validData), numpy.shape(validLabel),
          numpy.shape(testData), numpy.shape(testLabel))
    trainData.extend(validData)
    trainLabel.extend(validLabel)

    trainDataset = Dataset_General(data=trainData, label=trainLabel)
    testDataset = Dataset_General(data=testData, label=testLabel)

    return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True,
                                       collate_fn=Collate_FluctuateLen()), \
           torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                       collate_fn=Collate_FluctuateLen())


if __name__ == '__main__':
    trainDataset, testDataset = Loader_Audio()
    for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(trainDataset):
        print(numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
        print(batchLabel)
        exit()
