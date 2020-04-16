import os
import shutil

if __name__ == '__main__':
    loadPath = '/home/bztbztbzt/CMU-MOSEI-Result/AttentionMiddleState/'
    savePath = '/home/bztbztbzt/CMU-MOSEI-Result/AttentionMiddleState-Current/'
    for foldName in os.listdir(loadPath):
        for partName in os.listdir(os.path.join(loadPath, foldName)):
            print(foldName, partName)
            if partName.find('TestResult') != -1: continue
            os.makedirs(os.path.join(savePath, foldName, partName))
            shutil.copy(os.path.join(loadPath, foldName, partName, 'TrainMiddle-0099.npy'),
                        os.path.join(savePath, foldName, partName, 'TrainMiddle-0099.npy'))
            shutil.copy(os.path.join(loadPath, foldName, partName, 'TestMiddle-0099.npy'),
                        os.path.join(savePath, foldName, partName, 'TestMiddle-0099.npy'))
