from Auxiliary.Loader import Loader_Audio
from Model.BLSTMwAttention import BLSTMwAttention
from Auxiliary.TrainTemplate import TrainTemplate_FluctuateLength

if __name__ == '__main__':
    cudaFlag = True
    trainDataset, testDataset = Loader_Audio(batchSize=16, multiply=10)
    Model = BLSTMwAttention(attentionName='StandardAttention', attentionScope=0, featuresNumber=40, classNumber=6,
                            cudaFlag=cudaFlag)
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI_Audio/%s' % Model.moduleName
    TrainTemplate_FluctuateLength(Model=Model, trainDataset=trainDataset, testDataset=testDataset, savePath=savePath,
                                  cudaFlag=cudaFlag, saveFlag=True)
