from Auxiliary.Loader import Loader_Audio
from Auxiliary.TrainTemplate import TrainTemplate_FluctuateLength_Classification
from Model.BLSTMwAttention_Classification import BLSTMwAttention_ClassificationSingle

if __name__ == '__main__':
    cudaFlag = True
    appointEmotion = 1
    trainDataset, testDataset = Loader_Audio(batchSize=16, multiply=10, appointEmotion=appointEmotion)
    Model = BLSTMwAttention_ClassificationSingle(attentionName='StandardAttention', attentionScope=0, featuresNumber=40,
                                                 classNumber=2, cudaFlag=cudaFlag)
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI_Audio/Emotion%d/%s' % (appointEmotion, Model.moduleName)

    TrainTemplate_FluctuateLength_Classification(
        Model=Model, weight=[2, 0.5], trainDataset=trainDataset, testDataset=testDataset, savePath=savePath,
        cudaFlag=cudaFlag, saveFlag=True)
