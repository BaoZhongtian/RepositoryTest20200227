import os
import numpy

if __name__ == '__main__':
    loadPath = r'D:\PythonProjects_Data\CMU_MOSEI\TextPart\Step4_SeparateFold'
    for fold in os.listdir(loadPath):
        for fileName in os.listdir(os.path.join(loadPath, fold)):
            data = numpy.genfromtxt(fname=os.path.join(loadPath, fold, fileName), dtype=int, delimiter=',')
            print(data)
            print(fold, fileName)
            print(numpy.min(data))
            exit()
