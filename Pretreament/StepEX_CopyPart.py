import os
import shutil

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/AudioPart/Step4_SpectrumGeneration/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/AudioPart/Step4EX_SpectrumGeneration/'
    comparePath = 'D:/PythonProjects_Data/CMU_MOSEI/TextPart/Step4_SeparateFold/'

    for part in os.listdir(comparePath):
        if not os.path.exists(os.path.join(savePath, part)): os.makedirs(os.path.join(savePath, part))
        for fileName in os.listdir(os.path.join(comparePath, part)):
            # compareFileName = fileName[0:-7] + fileName[-6:]
            compareFileName = fileName
            # print(compareFileName)
            # exit()

            if not os.path.exists(os.path.join(loadPath, part, compareFileName)):
                print(part, fileName)
                continue
            shutil.copy(os.path.join(loadPath, part, compareFileName), os.path.join(savePath, part, compareFileName))
