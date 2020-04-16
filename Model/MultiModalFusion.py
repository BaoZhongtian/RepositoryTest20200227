import torch
import numpy
from Model.AttentionBase import AttentionBase
from Auxiliary.Loader import Loader_MultiModal_Single


class MultiModalFusion_SingleModal(AttentionBase):
    def __init__(self, attentionName, attentionScope, featuresNumber, classNumber, cudaFlag):
        super(MultiModalFusion_SingleModal, self).__init__(
            attentionName=attentionName, attentionScope=attentionScope, featuresNumber=256, cudaFlag=cudaFlag)
        self.moduleName = 'MMF-Single-%s-%d' % (attentionName, attentionScope)
        self.indexScope = int(featuresNumber / 256)
        self.firstNetwork = {}
        for index in range(int(featuresNumber / 256) + 1):
            self.firstNetwork['Layer1_Part%02d' % index] = self.__SimpleNNStructure()
        self.predict = torch.nn.Linear(in_features=256, out_features=classNumber)

    def __SimpleNNStructure(self):
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=256)
        )

    def forward(self, inputData, inputSeqLen=None):
        partResult = []
        for index in range(self.indexScope + 1):
            currentPart = self.firstNetwork['Layer1_Part%02d' % index](inputData[:, 256 * index:256 * (index + 1)])
            currentPart = currentPart.unsqueeze(1)
            # if self.cudaFlag: currentPart = currentPart.cuda()
            partResult.append(currentPart)
        partResult = torch.cat(partResult, dim=1)
        if self.cudaFlag: partResult = partResult.cuda()
        attentionResult, attentionHotMap = self.ApplyAttention(
            dataInput=partResult, attentionName=self.attentionName, inputSeqLen=None, hiddenNoduleNumbers=256)
        predict = self.predict(input=attentionResult)
        return predict, attentionHotMap, attentionResult


if __name__ == '__main__':
    trainDataset, testDataset = Loader_MultiModal_Single(
        appointPart=['Audio-1/BLSTM-W-StandardAttention-10', 'Audio-1/BLSTM-W-LocalAttention-10',
                     'Audio-1/BLSTM-W-MonotonicAttention-10'])
    model = MultiModalFusion_SingleModal(attentionName='StandardAttention', attentionScope=10, featuresNumber=256,
                                         classNumber=2, cudaFlag=True)
    model.cuda()
    for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(trainDataset):
        print(numpy.shape(batchData), numpy.shape(batchLabel))
        result, _, _ = model(batchData)
        # result = result.detach()
        print(numpy.shape(result))
        # for sample in result:
        #     print(numpy.shape(sample))
        exit()
