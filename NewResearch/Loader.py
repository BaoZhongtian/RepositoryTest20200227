import torch
import torch.utils.data as torch_utils_data
import numpy


class Dataset_Movielens(torch_utils_data.Dataset):
    def __init__(self, userData, movieData, rankData):
        self.userData, self.movieData, self.rankData = userData, movieData, rankData

    def __len__(self):
        return len(self.userData)

    def __getitem__(self, index):
        return self.userData[index], self.movieData[index], self.rankData[index]


class Collate_MovieLens:
    def __init__(self): pass

    def __call__(self, batch):
        xs = torch.LongTensor([v[0] for v in batch])
        ys = torch.LongTensor([v[1] for v in batch])
        zs = torch.IntTensor([v[2] for v in batch])
        return xs, ys, zs


def Loader_MovieLens_RandomSelect():
    loadPath = 'D:/PythonProjects/RepositoryTest20200227/NewResearch/Pretreatment/RecordData_Supplement.csv'
    totalData = numpy.genfromtxt(fname=loadPath, dtype=int, delimiter='::')
    numpy.random.shuffle(totalData)

    # for sample in totalData[0:10]: print(sample)
    userData = totalData[:, 0:4]
    movieData = numpy.concatenate([numpy.reshape(totalData[:, 4], [-1, 1]), totalData[:, 6:-1]], axis=1)
    rankData = totalData[:, -1]
    print(numpy.shape(totalData), numpy.shape(userData), numpy.shape(movieData), numpy.shape(rankData))
    # print(numpy.max(movieData, axis=0))

    trainDataset = Dataset_Movielens(
        userData=userData[0:int(0.8 * len(totalData))], movieData=movieData[0:int(0.8 * len(totalData))],
        rankData=rankData[0:int(0.8 * len(totalData))])
    testDataset = Dataset_Movielens(
        userData=userData[int(0.8 * len(totalData)):0], movieData=movieData[int(0.8 * len(totalData)):0],
        rankData=rankData[int(0.8 * len(totalData)):0])

    return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=64, shuffle=True,
                                       collate_fn=Collate_MovieLens()), \
           torch_utils_data.DataLoader(dataset=testDataset, batch_size=64, shuffle=False,
                                       collate_fn=Collate_MovieLens())


if __name__ == '__main__':
    trainDataset, testDataset = Loader_MovieLens_RandomSelect()
    for batchNumber, (userData, movieData, rankData) in enumerate(trainDataset):
        print(batchNumber, numpy.shape(userData), numpy.shape(movieData), numpy.shape(rankData))
