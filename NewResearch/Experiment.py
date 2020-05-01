import os
import torch
from NewResearch.Model.DoubleHand import DoubleHand
from NewResearch.Loader import Loader_MovieLens_RandomSelect

cudaFlag = True
learningRate = 1E-3

if __name__ == '__main__':
    savePath = 'D:/PythonProjects_Data/MovieLens/'
    if not os.path.exists(savePath): os.makedirs(savePath)

    Model = DoubleHand()
    if cudaFlag:
        Model.cuda()
        Model.cudaPretreatment()
    optimizer = torch.optim.Adam(params=Model.parameters(), lr=learningRate)
    lossFunction = torch.nn.CrossEntropyLoss()

    trainDataset, testDataset = Loader_MovieLens_RandomSelect()

    for episode in range(100):
        with open(os.path.join(savePath, 'Loss-%04d.csv' % episode), 'w') as file:
            episodeLoss = 0.0
            for batchNumber, (userData, movieData, rankData) in enumerate(trainDataset):
                if cudaFlag:
                    userData = userData.cuda()
                    movieData = movieData.cuda()
                    rankData = rankData.cuda()
                result = Model(userData=userData, movieData=movieData)
                loss = lossFunction(input=result, target=rankData.long()).cpu()

                file.write(str(loss.data.numpy()) + '\n')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('\rTraining %d Loss = %f' % (batchNumber, loss.data.numpy()), end='')
                episodeLoss += loss.data.numpy()
            print('\nEpisode %d Total Loss = %f' % (episode, episodeLoss))
