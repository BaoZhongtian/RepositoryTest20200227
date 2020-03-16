import torch
import numpy
from Auxiliary.Loader import Loader_Audio
from Model.AttentionBase import AttentionBase


class BLSTMwAttention_Regression(AttentionBase):
    def __init__(self, attentionName, attentionScope, featuresNumber, classNumber, cudaFlag):
        super(BLSTMwAttention_Regression, self).__init__(
            attentionName=attentionName, attentionScope=attentionScope, featuresNumber=128 * 2, cudaFlag=cudaFlag)
        self.moduleName = 'BLSTM-W-%s-%d' % (attentionName, attentionScope)
        self.rnnLayer = torch.nn.LSTM(input_size=featuresNumber, hidden_size=128, num_layers=2, bidirectional=True)
        self.predict = torch.nn.Linear(in_features=256, out_features=classNumber)

    def forward(self, inputData, inputSeqLen):
        inputData = inputData.float()
        rnnOutput, _ = self.rnnLayer(input=inputData, hx=None)
        attentionResult, attentionHotMap = self.ApplyAttention(
            dataInput=rnnOutput, attentionName=self.attentionName, inputSeqLen=inputSeqLen, hiddenNoduleNumbers=256)
        predict = self.predict(input=attentionResult)
        return predict, attentionHotMap


if __name__ == '__main__':
    trainDataset, testDataset = Loader_Audio()
    Model = BLSTMwAttention_Regression(
        attentionName='StandardAttention', attentionScope=0, featuresNumber=40, classNumber=6, cudaFlag=False)
    print(Model)
    for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(trainDataset):
        print(batchNumber, numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
        predict, attentionMap = Model(inputData=batchData, inputSeqLen=batchSeq)
        print(predict)
        print(numpy.shape(predict))

        exit()
