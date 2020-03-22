import os
import torch
import numpy
from Auxiliary.Tools import SaveNetwork


def TrainTemplate_FluctuateLength_Regression(Model, trainDataset, testDataset, savePath, cudaFlag=True, saveFlag=True,
                                             learningRate=1E-3, episodeNumber=100):
    if os.path.exists(savePath): return
    os.makedirs(savePath)
    os.makedirs(savePath + '-TestResult')

    print(Model)
    if cudaFlag: Model.cuda()
    optimizer = torch.optim.Adam(params=Model.parameters(), lr=learningRate)
    lossFunction = torch.nn.SmoothL1Loss()

    for episode in range(episodeNumber):
        episodeLoss = 0.0
        with open(os.path.join(savePath, 'Loss-%04d.csv' % episode), 'w') as file:
            for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(trainDataset):
                if cudaFlag:
                    batchData = batchData.cuda()
                    batchSeq = batchSeq.cuda()
                    batchLabel = batchLabel.cuda()
                # print(episode, numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
                result, attentionHotMap = Model(inputData=batchData, inputSeqLen=batchSeq)
                loss = lossFunction(input=result, target=batchLabel).cpu()

                file.write(str(loss.data.numpy()) + '\n')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episodeLoss += loss.data.numpy()
                print('\rTraining %d Loss = %f' % (batchNumber, loss.data.numpy()), end='')
        print('\nEpisode %d Total Loss = %f' % (episode, episodeLoss))

        #######################################################

        if saveFlag: SaveNetwork(model=Model, optimizer=optimizer,
                                 savePath=os.path.join(savePath, 'Episode-%04d' % episode))

        #######################################################

        testPredict, testLabel = [], []
        for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(testDataset):
            if cudaFlag:
                batchData = batchData.cuda()
                batchSeq = batchSeq.cuda()
            testLabel.extend(batchLabel.numpy())

            result, _ = Model(inputData=batchData, inputSeqLen=batchSeq)
            result = result.cpu().detach().numpy()
            testPredict.extend(result)
            print('\rTesting %d' % batchNumber, end='')

        with open(os.path.join(savePath + '-TestResult', 'Result-%04d.csv' % episode), 'w') as file:
            for indexX in range(len(testPredict)):
                for indexY in range(len(testPredict[indexX])):
                    if indexY != 0: file.write(',')
                    file.write(str(testPredict[indexX][indexY]))
                for indexY in range(len(testLabel[indexX])):
                    file.write(',' + str(testLabel[indexX][indexY]))
                file.write('\n')
        print('\nL1 Distance = %.2f' % (numpy.average(numpy.abs(numpy.subtract(testPredict, testLabel)))))


def TrainTemplate_FluctuateLength_Classification(Model, trainDataset, testDataset, savePath, weight=None, cudaFlag=True,
                                                 saveFlag=True, learningRate=1E-3, episodeNumber=100):
    if os.path.exists(savePath): return
    os.makedirs(savePath)
    os.makedirs(savePath + '-TestResult')

    print(Model)
    if cudaFlag: Model.cuda()
    optimizer = torch.optim.Adam(params=Model.parameters(), lr=learningRate)

    if weight is not None:
        weightTensor = torch.FloatTensor(weight)
        if cudaFlag: weightTensor = weightTensor.cuda()
        lossFunction = torch.nn.CrossEntropyLoss(weight=weightTensor)
    else:
        lossFunction = torch.nn.CrossEntropyLoss()

    for episode in range(episodeNumber):
        episodeLoss = 0.0
        with open(os.path.join(savePath, 'Loss-%04d.csv' % episode), 'w') as file:
            for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(trainDataset):
                if cudaFlag:
                    batchData = batchData.cuda()
                    batchSeq = batchSeq.cuda()
                    batchLabel = batchLabel.cuda()
                # print(episode, numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
                result, attentionHotMap = Model(inputData=batchData, inputSeqLen=batchSeq)
                loss = lossFunction(input=result, target=batchLabel.long()).cpu()

                file.write(str(loss.data.numpy()) + '\n')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episodeLoss += loss.data.numpy()
                print('\rTraining %d Loss = %f' % (batchNumber, loss.data.numpy()), end='')
        print('\nEpisode %d Total Loss = %f' % (episode, episodeLoss))

        #######################################################

        if saveFlag: SaveNetwork(model=Model, optimizer=optimizer,
                                 savePath=os.path.join(savePath, 'Episode-%04d' % episode))

        #######################################################

        testPredict, testLabel = [], []
        for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(testDataset):
            if cudaFlag:
                batchData = batchData.cuda()
                batchSeq = batchSeq.cuda()
            testLabel.extend(batchLabel.numpy())

            result, _ = Model(inputData=batchData, inputSeqLen=batchSeq)
            result = result.cpu().detach().numpy()
            testPredict.extend(result)
            print('\rTesting %d' % batchNumber, end='')

        with open(os.path.join(savePath + '-TestResult', 'Result-%04d.csv' % episode), 'w') as file:
            for indexX in range(len(testPredict)):
                for indexY in range(len(testPredict[indexX])):
                    file.write(str(testPredict[indexX][indexY]) + ',')
                file.write(str(testLabel[indexX]) + '\n')

        # print('\nPrecision = %.2f' % numpy.sum(testLabel == numpy.argmax(testPredict, axis=1)) / len(testLabel) * 100)
