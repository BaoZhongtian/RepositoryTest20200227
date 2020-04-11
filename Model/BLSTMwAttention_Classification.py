import torch
import numpy
from Auxiliary.Loader import Loader_Text
from Model.AttentionBase import AttentionBase


class BLSTMwAttention_ClassificationSingle(AttentionBase):
    def __init__(self, attentionName, attentionScope, featuresNumber, classNumber, cudaFlag):
        super(BLSTMwAttention_ClassificationSingle, self).__init__(
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
        return predict, attentionHotMap, attentionResult


class BLSTMwAttention_ClassificationSingle_Text(AttentionBase):
    def __init__(self, attentionName, attentionScope, featuresNumber, classNumber, cudaFlag):
        super(BLSTMwAttention_ClassificationSingle_Text, self).__init__(
            attentionName=attentionName, attentionScope=attentionScope, featuresNumber=128 * 2, cudaFlag=cudaFlag)
        self.moduleName = 'BLSTM-W-%s-%d' % (attentionName, attentionScope)
        self.embeddingLayer = torch.nn.Embedding(num_embeddings=20000, embedding_dim=128)
        self.rnnLayer = torch.nn.LSTM(input_size=featuresNumber, hidden_size=128, num_layers=2, bidirectional=True)
        self.predict = torch.nn.Linear(in_features=256, out_features=classNumber)

    def forward(self, inputData, inputSeqLen):
        inputData = inputData.long()
        embeddingData = self.embeddingLayer(input=inputData)
        rnnOutput, _ = self.rnnLayer(input=embeddingData, hx=None)
        attentionResult, attentionHotMap = self.ApplyAttention(
            dataInput=rnnOutput, attentionName=self.attentionName, inputSeqLen=inputSeqLen, hiddenNoduleNumbers=256)
        predict = self.predict(input=attentionResult)
        return predict, attentionHotMap, attentionResult


if __name__ == '__main__':
    trainDataset, testDataset = Loader_Text()
    Model = BLSTMwAttention_ClassificationSingle_Text(
        attentionName='StandardAttention', attentionScope=0, featuresNumber=128, classNumber=2, cudaFlag=False)
    print(Model)
    for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(trainDataset):
        print(batchNumber, numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
        predict, _ = Model(inputData=batchData, inputSeqLen=batchSeq)
        print(predict)
        print(numpy.shape(predict))

        exit()
