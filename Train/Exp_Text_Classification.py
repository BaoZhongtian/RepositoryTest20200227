from Auxiliary.Loader import Loader_Text
from Auxiliary.TrainTemplate import TrainTemplate_FluctuateLength_Classification
from Model.BLSTMwAttention_Classification import BLSTMwAttention_ClassificationSingle_Text

if __name__ == '__main__':
    cudaFlag = True
    trainDataset, testDataset = Loader_Text(batchSize=16)
    for attentionName in ['StandardAttention', 'LocalAttention', 'ComponentAttention', 'MonotonicAttention']:
        Model = BLSTMwAttention_ClassificationSingle_Text(attentionName=attentionName, attentionScope=10,
                                                          featuresNumber=128, classNumber=2, cudaFlag=cudaFlag)
        savePath = 'D:/PythonProjects_Data/CMU_MOSEI_Text/%s' % Model.moduleName

        TrainTemplate_FluctuateLength_Classification(
            Model=Model, trainDataset=trainDataset, testDataset=testDataset, savePath=savePath,
            cudaFlag=cudaFlag, saveFlag=True)
