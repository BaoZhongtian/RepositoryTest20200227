import os
import shutil

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/VideoPart/Step3_SeparateFold/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/VideoPart/Step3_SeparateFold_EX/'
    for foldName in os.listdir(loadPath):
        if not os.path.exists(os.path.join(savePath, foldName)): os.makedirs(os.path.join(savePath, foldName))
        for fileName in os.listdir(os.path.join(loadPath, foldName)):
            saveName = fileName[0:-6] + '_' + fileName[-6:]
            print(foldName, saveName)
            shutil.copy(os.path.join(loadPath, foldName, fileName), os.path.join(savePath, foldName, saveName))

            # exit()
