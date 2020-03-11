import os
import numpy
from pydub import AudioSegment

if __name__ == '__main__':
    audioPath = 'D:/PythonProjects_Data/CMU_MOSEI/WAV_16000/'
    labelPath = 'D:/PythonProjects_Data/CMU_MOSEI/Step1_StartEndCut_Emotion/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/Step2_AudioCut/'
    if not os.path.exists(savePath): os.makedirs(savePath)

    for fileName in os.listdir(labelPath):
        print('Treating', fileName)
        startEndTicket = numpy.reshape(
            numpy.genfromtxt(fname=os.path.join(labelPath, fileName), dtype=float, delimiter=','), [-1, 8])
        wavFile = AudioSegment.from_file(file=os.path.join(audioPath, fileName.replace('csv', 'wav')))

        for counter in range(numpy.shape(startEndTicket)[0]):
            wavFile[int(startEndTicket[counter][-2] * 1000):int(startEndTicket[counter][-1] * 1000)].export(
                os.path.join(savePath, fileName.replace('.csv', '_%02d.wav' % counter)), format='wav')

        # exit()
