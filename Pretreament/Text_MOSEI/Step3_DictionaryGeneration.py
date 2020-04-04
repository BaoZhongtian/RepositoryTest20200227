import os

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/CMU_MOSEI/TextPart/Step1_ChoosePart/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/TextPart/Step3_DictionaryGeneration/'
    dictionaryPath = 'D:/PythonProjects_Data/CMU_MOSEI/TextPart/Dictionary.csv'

    if not os.path.exists(savePath): os.makedirs(savePath)
    dictionary = {}

    for fileName in os.listdir(loadPath):
        with open(os.path.join(loadPath, fileName), 'r') as file:
            data = file.read()
        data = data[0:-1].upper()

        treatedData = ''
        for sample in data:
            if sample == ' ' or ('A' <= sample <= 'Z'): treatedData += sample
        treatedData = treatedData.split(' ')
        for sample in treatedData:
            if sample not in dictionary.keys():
                dictionary[sample] = len(dictionary.keys())

        # exit()
    with open(dictionaryPath, 'w') as file:
        for sample in dictionary.keys():
            file.write(sample + ',' + str(dictionary[sample]) + '\n')

    ##############################
    print('Dictionary Completed')
    ##############################

    for fileName in os.listdir(loadPath):
        with open(os.path.join(loadPath, fileName), 'r') as file:
            data = file.read()
        data = data[0:-1].upper()

        treatedData = ''
        for sample in data:
            if sample == ' ' or ('A' <= sample <= 'Z'): treatedData += sample
        treatedData = treatedData.split(' ')

        with open(os.path.join(savePath, fileName.replace('txt', 'csv')), 'w') as file:
            for sample in treatedData:
                file.write(str(dictionary[sample]) + ',')
