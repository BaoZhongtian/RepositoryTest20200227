import torch
import numpy
from NewResearch.Loader import Loader_MovieLens_RandomSelect


class DoubleHand(torch.nn.Module):
    def __init__(self):
        super(DoubleHand, self).__init__()
        self.userEmbedding = torch.nn.Embedding(num_embeddings=6050, embedding_dim=16)
        self.genderEmbedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=16)
        self.ageEmbedding = torch.nn.Embedding(num_embeddings=7, embedding_dim=16)
        self.occupationEmbedding = torch.nn.Embedding(num_embeddings=25, embedding_dim=16)
        self.movieEmbedding = torch.nn.Embedding(num_embeddings=4000, embedding_dim=16)
        self.movieTypeEmbedding = []
        for _ in range(18): self.movieTypeEmbedding.append(torch.nn.Embedding(num_embeddings=2, embedding_dim=4))
        # self.userLayer1st=torch.nn.Linear()
        self.userLayer1st = torch.nn.Linear(in_features=64, out_features=128)
        self.userLayer2nd = torch.nn.Linear(in_features=128, out_features=128)
        self.movieLayer1st = torch.nn.Linear(in_features=88, out_features=128)
        self.movieLayer2nd = torch.nn.Linear(in_features=128, out_features=128)

        self.predictLayer = torch.nn.Linear(in_features=128, out_features=6)

    def forward(self, userData, movieData):
        userEmbeddingResult = self.userEmbedding(userData[:, 0])
        genderEmbeddingResult = self.genderEmbedding(userData[:, 1])
        ageEmbeddingResult = self.ageEmbedding(userData[:, 2])
        occupationEmbeddingResult = self.occupationEmbedding(userData[:, 3])

        userInformation = torch.cat(
            [userEmbeddingResult, genderEmbeddingResult, ageEmbeddingResult, occupationEmbeddingResult], dim=1)
        user1st = self.userLayer1st(userInformation).relu()
        userResult = self.userLayer2nd(user1st)

        movieEmbeddingResult = self.movieEmbedding(movieData[:, 0])
        movieInformation = [movieEmbeddingResult]
        for index in range(18):
            movieInformation.append(self.movieTypeEmbedding[index](movieData[:, index + 1]))
        movieInformation = torch.cat(movieInformation, dim=1)
        movie1st = self.movieLayer1st(movieInformation).relu()
        movieResult = self.movieLayer2nd(movie1st)

        assemblyResult = torch.mul(userResult, movieResult)
        predict = self.predictLayer(input=assemblyResult)

        return predict

    def cudaPretreatment(self):
        for sample in self.movieTypeEmbedding:
            sample.cuda()


if __name__ == '__main__':
    Model = DoubleHand()
    trainDataset, testDataset = Loader_MovieLens_RandomSelect()
    for batchNumber, (userData, movieData, rankData) in enumerate(trainDataset):
        print(batchNumber, numpy.shape(userData), numpy.shape(movieData), numpy.shape(rankData))
        result = Model(userData=userData, movieData=movieData)
        print(numpy.shape(result))
        exit()
