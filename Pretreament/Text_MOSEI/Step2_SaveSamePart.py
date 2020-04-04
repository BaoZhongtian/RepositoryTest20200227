import os
import shutil

if __name__ == '__main__':
    transcriptPath = 'D:/PythonProjects_Data/CMU_MOSEI/TextPart/Step1_ChoosePart/'
    audioPath = 'D:/PythonProjects_Data/CMU_MOSEI/AudioPart/Step2_AudioCut/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/TextPart/Step2_SaveSamePart/'

    for fileName in os.listdir(transcriptPath):
        print(fileName)
        if not os.path.exists(os.path.join(audioPath, fileName.replace('txt', 'wav'))): print(fileName)
        # exit()
