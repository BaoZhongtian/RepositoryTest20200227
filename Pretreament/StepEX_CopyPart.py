import os
import shutil

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/VideoPart/Step3_SeparateFold_EX/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/VideoPart/Step3SAME_SeparateFold_EX/'
    comparePathA = 'D:/PythonProjects_Data/CMU_MOSEI/AudioPart/Step4_SpectrumGeneration/'
    comparePathB = 'D:/PythonProjects_Data/CMU_MOSEI/TextPart/Step4_SeparateFold/'

    for foldName in os.listdir(loadPath):
        if not os.path.exists(os.path.join(savePath, foldName)): os.makedirs(os.path.join(savePath, foldName))
        counter = 0
        for fileName in os.listdir(os.path.join(loadPath, foldName)):
            if (not os.path.exists(os.path.join(comparePathA, foldName, fileName))) or (
                    not os.path.exists(os.path.join(comparePathB, foldName, fileName))):
                # print(foldName, fileName)
                counter += 1
                continue
            shutil.copy(os.path.join(loadPath, foldName, fileName), os.path.join(savePath, foldName, fileName))
        print(foldName, counter)
