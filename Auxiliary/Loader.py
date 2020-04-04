import os
import torch
import numpy
from torch.utils import data as torch_utils_data


class Collate_FluctuateLen_Regression:
    def __init__(self, labelPadding=False, dimensionOneFlag=False):
        self.labelPadding = labelPadding
        self.dimensionOneFlag = dimensionOneFlag

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
        if self.dimensionOneFlag:
            xs = numpy.array([self.__LabelTensor(dataInput=v, maxLen=max_len) for v in xs], dtype=float)
        else:
            xs = numpy.array([self.__DataTensor(dataInput=v, maxLen=max_len) for v in xs], dtype=float)
        xs = torch.from_numpy(xs)
        return xs, seq_lengths, ys

    def __call__(self, batch):
        return self.__collateOnlyData(batch=batch)


class Collate_FluctuateLen_Classification:
    def __init__(self, labelPadding=False):
        self.labelPadding = labelPadding

    def __DataTensor(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros([maxLen - len(dataInput), numpy.shape(dataInput)[1]],
                                                         dtype=torch.float)], axis=0)

    def __LabelTensor(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros([maxLen - len(dataInput)], dtype=torch.float)], axis=0)

    def __collateOnlyData(self, batch):
        xs = [v[0] for v in batch]
        ys = torch.IntTensor([v[1] for v in batch])
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


###########################################################

class Collate_AttentionTransform:
    def __init__(self, labelPadding=False):
        self.labelPadding = labelPadding

    def __DataTensor(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros([maxLen - len(dataInput), numpy.shape(dataInput)[1]],
                                                         dtype=torch.float)], axis=0)

    def __AttentionMapTensor(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros(maxLen - len(dataInput), dtype=torch.float)], axis=0)

    def __LabelTensor(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros([maxLen - len(dataInput)], dtype=torch.float)], axis=0)

    def __collateOnlyData(self, batch):
        currentData = [v[0] for v in batch]
        currentLabel = torch.IntTensor([v[1] for v in batch])
        currentMap = [v[2] for v in batch]

        seq_lengths = torch.LongTensor([v for v in map(len, currentData)])
        max_len = max([len(v) for v in currentData])
        currentData = numpy.array([self.__DataTensor(dataInput=v, maxLen=max_len) for v in currentData], dtype=float)
        currentData = torch.from_numpy(currentData)

        max_len = max([len(v) for v in currentMap])
        currentMap = numpy.array([self.__AttentionMapTensor(dataInput=v, maxLen=max_len) for v in currentMap],
                                 dtype=float)
        currentMap = torch.from_numpy(currentMap)

        return currentData, seq_lengths, currentLabel, currentMap

    def __call__(self, batch):
        return self.__collateOnlyData(batch=batch)


class Dataset_AttentionTransform(torch_utils_data.Dataset):
    def __init__(self, data, label, attentionHotMap):
        self.data, self.label, self.attentionHotMap = data, label, attentionHotMap

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.attentionHotMap[index]


###########################################################


def Loader_Audio(batchSize=32):
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/Data_Audio/'
    trainData = numpy.load(file=os.path.join(loadPath, 'Train-Data.npy'), allow_pickle=True).tolist()
    trainLabel = numpy.load(file=os.path.join(loadPath, 'Train-Label.npy'), allow_pickle=True).tolist()
    validData = numpy.load(file=os.path.join(loadPath, 'Valid-Data.npy'), allow_pickle=True).tolist()
    validLabel = numpy.load(file=os.path.join(loadPath, 'Valid-Label.npy'), allow_pickle=True).tolist()
    testData = numpy.load(file=os.path.join(loadPath, 'Test-Data.npy'), allow_pickle=True).tolist()
    testLabel = numpy.load(file=os.path.join(loadPath, 'Test-Label.npy'), allow_pickle=True).tolist()

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(validData), numpy.shape(validLabel),
          numpy.shape(testData), numpy.shape(testLabel))
    print(numpy.sum(trainLabel), numpy.sum(validLabel), numpy.sum(testLabel))

    trainData.extend(validData)
    trainLabel.extend(validLabel)

    trainDataset = Dataset_General(data=trainData, label=trainLabel)
    testDataset = Dataset_General(data=testData, label=testLabel)
    return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True,
                                       collate_fn=Collate_FluctuateLen_Regression()), \
           torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                       collate_fn=Collate_FluctuateLen_Regression())


def Loader_Text(batchSize=32):
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/Data_Text/'
    trainData = numpy.load(file=os.path.join(loadPath, 'Train-Data.npy'), allow_pickle=True).tolist()
    trainLabel = numpy.load(file=os.path.join(loadPath, 'Train-Label.npy'), allow_pickle=True).tolist()
    validData = numpy.load(file=os.path.join(loadPath, 'Valid-Data.npy'), allow_pickle=True).tolist()
    validLabel = numpy.load(file=os.path.join(loadPath, 'Valid-Label.npy'), allow_pickle=True).tolist()
    testData = numpy.load(file=os.path.join(loadPath, 'Test-Data.npy'), allow_pickle=True).tolist()
    testLabel = numpy.load(file=os.path.join(loadPath, 'Test-Label.npy'), allow_pickle=True).tolist()

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(validData), numpy.shape(validLabel),
          numpy.shape(testData), numpy.shape(testLabel))
    print(numpy.sum(trainLabel), numpy.sum(validLabel), numpy.sum(testLabel))

    trainData.extend(validData)
    trainLabel.extend(validLabel)

    trainDataset = Dataset_General(data=trainData, label=trainLabel)
    testDataset = Dataset_General(data=testData, label=testLabel)
    return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True,
                                       collate_fn=Collate_FluctuateLen_Regression(dimensionOneFlag=True)), \
           torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                       collate_fn=Collate_FluctuateLen_Regression(dimensionOneFlag=True))


def Loader_Video(appointPart, batchSize=32):
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/Data_Video_%s/' % appointPart
    trainData = numpy.load(file=os.path.join(loadPath, 'Train-Data.npy'), allow_pickle=True).tolist()
    trainLabel = numpy.load(file=os.path.join(loadPath, 'Train-Label.npy'), allow_pickle=True).tolist()
    validData = numpy.load(file=os.path.join(loadPath, 'Valid-Data.npy'), allow_pickle=True).tolist()
    validLabel = numpy.load(file=os.path.join(loadPath, 'Valid-Label.npy'), allow_pickle=True).tolist()
    testData = numpy.load(file=os.path.join(loadPath, 'Test-Data.npy'), allow_pickle=True).tolist()
    testLabel = numpy.load(file=os.path.join(loadPath, 'Test-Label.npy'), allow_pickle=True).tolist()

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(validData), numpy.shape(validLabel),
          numpy.shape(testData), numpy.shape(testLabel))
    print(numpy.sum(trainLabel), numpy.sum(validLabel), numpy.sum(testLabel))

    trainData.extend(validData)
    trainLabel.extend(validLabel)

    trainDataset = Dataset_General(data=trainData, label=trainLabel)
    testDataset = Dataset_General(data=testData, label=testLabel)
    return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True,
                                       collate_fn=Collate_FluctuateLen_Regression()), \
           torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                       collate_fn=Collate_FluctuateLen_Regression())


def Loader_Total(appointPart, batchSize=32):
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/Transform_Data_%s/' % appointPart
    trainData = numpy.load(file=os.path.join(loadPath, 'Train-Data.npy'), allow_pickle=True).tolist()
    trainLabel = numpy.load(file=os.path.join(loadPath, 'Train-Label.npy'), allow_pickle=True).tolist()
    validData = numpy.load(file=os.path.join(loadPath, 'Valid-Data.npy'), allow_pickle=True).tolist()
    validLabel = numpy.load(file=os.path.join(loadPath, 'Valid-Label.npy'), allow_pickle=True).tolist()
    testData = numpy.load(file=os.path.join(loadPath, 'Test-Data.npy'), allow_pickle=True).tolist()
    testLabel = numpy.load(file=os.path.join(loadPath, 'Test-Label.npy'), allow_pickle=True).tolist()

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(validData), numpy.shape(validLabel),
          numpy.shape(testData), numpy.shape(testLabel))
    print(numpy.sum(trainLabel), numpy.sum(validLabel), numpy.sum(testLabel))

    trainData.extend(validData)
    trainLabel.extend(validLabel)

    trainDataset = Dataset_General(data=trainData, label=trainLabel)
    testDataset = Dataset_General(data=testData, label=testLabel)
    return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True,
                                       collate_fn=Collate_FluctuateLen_Regression()), \
           torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=False,
                                       collate_fn=Collate_FluctuateLen_Regression()), \
           torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                       collate_fn=Collate_FluctuateLen_Regression())


def Loader_AttentionTransform(appointPart, appointAttention, batchSize=32):
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/Transform_Data_%s/' % appointPart
    trainData = numpy.load(file=os.path.join(loadPath, 'Train-Data.npy'), allow_pickle=True).tolist()
    trainLabel = numpy.load(file=os.path.join(loadPath, 'Train-Label.npy'), allow_pickle=True).tolist()
    validData = numpy.load(file=os.path.join(loadPath, 'Valid-Data.npy'), allow_pickle=True).tolist()
    validLabel = numpy.load(file=os.path.join(loadPath, 'Valid-Label.npy'), allow_pickle=True).tolist()
    testData = numpy.load(file=os.path.join(loadPath, 'Test-Data.npy'), allow_pickle=True).tolist()
    testLabel = numpy.load(file=os.path.join(loadPath, 'Test-Label.npy'), allow_pickle=True).tolist()

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(validData), numpy.shape(validLabel),
          numpy.shape(testData), numpy.shape(testLabel))
    print(numpy.sum(trainLabel), numpy.sum(validLabel), numpy.sum(testLabel))

    trainData.extend(validData)
    trainLabel.extend(validLabel)

    if appointPart == 'Audio': attentionPath = 'D:/PythonProjects_Data/AttentionHotMap/Video/%s/' % appointAttention
    if appointPart == 'Video': attentionPath = 'D:/PythonProjects_Data/AttentionHotMap/Audio/%s/' % appointAttention

    trainAttentionHotMap = numpy.load(file=os.path.join(attentionPath, 'TrainAttentionHotMap.npy'), allow_pickle=True)
    testAttentionHotMap = numpy.load(file=os.path.join(attentionPath, 'TestAttentionHotMap.npy'), allow_pickle=True)

    trainDataset = Dataset_AttentionTransform(data=trainData, label=trainLabel, attentionHotMap=trainAttentionHotMap)
    testDataset = Dataset_AttentionTransform(data=testData, label=testLabel, attentionHotMap=testAttentionHotMap)

    return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True,
                                       collate_fn=Collate_AttentionTransform()), \
           torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                       collate_fn=Collate_AttentionTransform())


if __name__ == '__main__':
    trainDataset, testDataset = Loader_Text()
    for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(trainDataset):
        print(numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
        # print(batchLabel)
        # exit()
