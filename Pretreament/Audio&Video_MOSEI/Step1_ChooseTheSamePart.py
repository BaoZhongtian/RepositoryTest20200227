import os
import shutil

if __name__ == '__main__':
    audioPath = 'D:/PythonProjects_Data/CMU_MOSEI/AudioPart/Step5_Normalization/'
    videoPath = 'D:/PythonProjects_Data/CMU_MOSEI/VideoPart/Step4_Normalization/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSEI/TotalPart/'

    # for part in os.listdir(videoPath):
    #     for fileName in os.listdir(os.path.join(videoPath, part)):
    #         print(part, fileName)
    #         newFileName = fileName[0:-6] + '_' + fileName[-6:]
    #         os.rename(os.path.join(videoPath, part, fileName), os.path.join(videoPath, part, newFileName))

    for part in os.listdir(audioPath):
        os.makedirs(os.path.join(savePath, 'Audio', part))
        for fileName in os.listdir(os.path.join(audioPath, part)):
            print('Audio', part, fileName)
            if os.path.exists(os.path.join(videoPath, part, fileName)):
                shutil.copy(os.path.join(audioPath, part, fileName), os.path.join(savePath, 'Audio', part, fileName))
    for part in os.listdir(videoPath):
        os.makedirs(os.path.join(savePath, 'Video', part))
        for fileName in os.listdir(os.path.join(videoPath, part)):
            print('Video', part, fileName)
            if os.path.exists(os.path.join(audioPath, part, fileName)):
                shutil.copy(os.path.join(videoPath, part, fileName), os.path.join(savePath, 'Video', part, fileName))
