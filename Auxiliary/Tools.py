import torch
import numpy


def UA_Calculation(matrix):
    return (matrix[0][0] / numpy.sum(matrix[0]) + matrix[1][1] / numpy.sum(matrix[1])) / 2


def F1Score_Calculation(matrix):
    precision = matrix[0][0] / numpy.sum(matrix[0])
    recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])
    if precision + recall == 0: return 0
    return 2 * precision * recall / (precision + recall)


def SaveNetwork(model, optimizer, savePath):
    torch.save(obj={'ModelStateDict': model.state_dict(), 'OptimizerStateDict': optimizer.state_dict()},
               f=savePath + '-Parameter.pkl')
    torch.save(obj=model, f=savePath + '-Network.pkl')


def LoadNetworkParameter(model, optimizer, loadPath):
    checkpoint = torch.load(loadPath)
    model.load_state_dict(checkpoint['ModelStateDict'])
    optimizer.load_state_dict(checkpoint['OptimizerStateDict'])
    return model, optimizer
