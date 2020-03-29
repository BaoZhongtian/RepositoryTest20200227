from Auxiliary.Loader import Loader_Total
from Auxiliary.TrainTemplate import TrainTemplate_AttentionMapGeneration
from Model.BLSTMwAttention_Classification import BLSTMwAttention_ClassificationSingle

if __name__ == '__main__':
    cudaFlag = True
    trainDataset, frozenTrainDataset, testDataset = Loader_Total(batchSize=16, appointPart='Audio')
    for attentionName in ['StandardAttention', 'LocalAttention', 'ComponentAttention', 'MonotonicAttention']:
        Model = BLSTMwAttention_ClassificationSingle(attentionName=attentionName, attentionScope=10,
                                                     featuresNumber=40, classNumber=2, cudaFlag=cudaFlag)
        savePath = 'D:/PythonProjects_Data/CMU_MOSEI_AttentionHotMap/Audio/%s' % Model.moduleName

        TrainTemplate_AttentionMapGeneration(
            Model=Model, trainDataset=trainDataset, frozenTrainDataset=frozenTrainDataset, testDataset=testDataset,
            savePath=savePath, cudaFlag=cudaFlag, saveFlag=True)
