import os
import numpy

loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/Data_Audio/'
trainLabel = numpy.load(file=os.path.join(loadPath, 'Train-Label.npy'), allow_pickle=True)

# counter = 0
# for sample in trainLabel:
#     if sample[3] < 0: counter += 1
# print(counter)
print(numpy.min(trainLabel, axis=0))
