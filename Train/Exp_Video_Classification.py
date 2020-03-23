from Auxiliary.Loader import Loader_Video
from Auxiliary.TrainTemplate import TrainTemplate_FluctuateLength_Classification
from Model.BLSTMwAttention_Classification import BLSTMwAttention_ClassificationSingle

if __name__ == '__main__':
    cudaFlag = True
    trainDataset, testDataset = Loader_Video(appointPart='Facet', batchSize=16)
    for attentionName in ['StandardAttention', 'LocalAttention', 'ComponentAttention', 'MonotonicAttention']:
        Model = BLSTMwAttention_ClassificationSingle(attentionName=attentionName, attentionScope=10,
                                                     featuresNumber=35, classNumber=2, cudaFlag=cudaFlag)
        savePath = 'D:/PythonProjects_Data/CMU_MOSEI_Video/%s' % Model.moduleName

        TrainTemplate_FluctuateLength_Classification(
            Model=Model, trainDataset=trainDataset, testDataset=testDataset, savePath=savePath,
            cudaFlag=cudaFlag, saveFlag=True)
