from Auxiliary.Loader import Loader_AttentionTransform
from Auxiliary.TrainTemplate import TrainTemplate_AttentionTransform
from Model.BLSTMwAttention_Classification import BLSTMwAttention_ClassificationSingle

if __name__ == '__main__':
    cudaFlag = True
    appointPart = 'Audio'
    for attentionWeight in [1, 5, 10, 50, 100]:
        for attentionName in ['StandardAttention', 'LocalAttention', 'ComponentAttention', 'MonotonicAttention']:
            trainDataset, testDataset = Loader_AttentionTransform(batchSize=16, appointPart=appointPart,
                                                                  appointAttention=attentionName)
            Model = BLSTMwAttention_ClassificationSingle(attentionName=attentionName, attentionScope=10,
                                                         featuresNumber=40, classNumber=2, cudaFlag=cudaFlag)

            savePath = 'D:/PythonProjects_Data/CMU_MOSEI_AttentionTransform/%s-%d/%s' % (
                appointPart, attentionWeight, Model.moduleName)

            TrainTemplate_AttentionTransform(
                Model=Model, trainDataset=trainDataset, testDataset=testDataset, savePath=savePath,
                cudaFlag=cudaFlag, saveFlag=False, attentionWeight=10)
