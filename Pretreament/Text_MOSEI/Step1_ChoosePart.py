import os
import numpy

if __name__ == '__main__':
    timeTicketPath = 'D:/PythonProjects_Data/CMU_MOSEI/Step1_StartEndCut/'
    transcriptPath = 'D:/PythonProjects_Data/CMU_MOSEI/Transcript-Raw/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/TextPart/Step1_ChoosePart/'
    if not os.path.exists(savePath): os.makedirs(savePath)

    for fileName in os.listdir(timeTicketPath):
        print(fileName)
        counter = 0
        timeTicketData = numpy.reshape(
            numpy.genfromtxt(fname=os.path.join(timeTicketPath, fileName), dtype=str, delimiter=','), [-1, 3])

        try:
            with open(os.path.join(transcriptPath, fileName.replace('csv', 'txt')), 'r') as file:
                transcriptData = file.readlines()
        except:
            continue

        for indexX in range(numpy.shape(timeTicketData)[0]):
            with open(os.path.join(savePath, fileName.replace('.csv', '_%02d.txt' % counter)), 'w') as file:
                for indexY in range(len(transcriptData)):
                    treatData = transcriptData[indexY].split('___')
                    if abs(float(timeTicketData[indexX][1]) - float(treatData[2])) <= 1 and \
                            abs(float(timeTicketData[indexX][2]) - float(treatData[3])) <= 1:
                        # print(treatData[4])
                        file.write(treatData[4])
                        continue
                counter += 1
        # exit()
