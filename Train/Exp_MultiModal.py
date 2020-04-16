from Auxiliary.Loader import Loader_MultiModal_Single
from Model.MultiModalFusion import MultiModalFusion_SingleModal
from Auxiliary.TrainTemplate import TrainTemplate_FluctuateLength_Classification

if __name__ == '__main__':
    for part in ['Audio', 'Video']:
        for scope in [1, 5, 10, 50, 100]:
            trainDataset, testDataset = Loader_MultiModal_Single(
                appointPart=['%s-%d/BLSTM-W-StandardAttention-10' % (part, scope),
                             '%s-%d/BLSTM-W-LocalAttention-10' % (part, scope),
                             '%s-%d/BLSTM-W-MonotonicAttention-10' % (part, scope)])
            Model = MultiModalFusion_SingleModal(attentionName='StandardAttention', attentionScope=10,
                                                 featuresNumber=256, classNumber=2, cudaFlag=True)
            # Model.cuda()
            for sample in Model.firstNetwork.keys():
                Model.firstNetwork[sample].cuda()

            savePath = 'D:/PythonProjects_Data/CMU_MOSEI_MultiFusion/%s-%d' % (part, scope)
            TrainTemplate_FluctuateLength_Classification(
                Model=Model, trainDataset=trainDataset, testDataset=testDataset, savePath=savePath,
                cudaFlag=True, saveFlag=False)
