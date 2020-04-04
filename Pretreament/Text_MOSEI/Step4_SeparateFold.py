import os
import shutil

if __name__ == '__main__':
    part = 'train'
    labelPath = 'D:/PythonProjects_Data/CMU_MOSEI/Metadata/standard_%s_fold/' % part
    dataPath = 'D:/PythonProjects_Data/CMU_MOSEI/TextPart/Step3_DictionaryGeneration/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/TextPart/Step4_SeparateFold/%s/' % part
    if not os.path.exists(savePath): os.makedirs(savePath)
    dictionary = set()
    for fileName in os.listdir(labelPath): dictionary.add(fileName.replace('.json', ''))

    for fileName in os.listdir(dataPath):
        if fileName[:-7] in dictionary:
            print(fileName)
            shutil.copy(os.path.join(dataPath, fileName), os.path.join(savePath, fileName))
    print(dictionary)
