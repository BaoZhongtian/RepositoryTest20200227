from Auxiliary.Loader import Loader_Audio
from Auxiliary.TrainTemplate import TrainTemplate_FluctuateLength_Classification
from Model.BLSTMwAttention_Classification import BLSTMwAttention_ClassificationSingle

if __name__ == '__main__':
    cudaFlag = True
    trainDataset, testDataset = Loader_Audio(batchSize=16)
    Model = BLSTMwAttention_ClassificationSingle(attentionName='StandardAttention', attentionScope=0, featuresNumber=40,
                                                 classNumber=2, cudaFlag=cudaFlag)
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI_Audio/%s' % Model.moduleName

    TrainTemplate_FluctuateLength_Classification(
        Model=Model, trainDataset=trainDataset, testDataset=testDataset, savePath=savePath,
        cudaFlag=cudaFlag, saveFlag=True)
