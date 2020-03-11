import os
import scipy
import numpy
import librosa

mBands = 40
sRate = 16000
winLength = int(0.025 * sRate)
hopLength = int(0.010 * sRate)
nFFT = winLength

if __name__ == '__main__':
    dataPath = 'D:/PythonProjects_Data/CMU_MOSI/Raw/Audio/WAV_16000/Segmented/'
    savePath = 'D:/PythonProjects_Data/CMU_MOSI/Step2_SpectrumGeneration/'

    # os.makedirs(savePath)
    for fileName in os.listdir(dataPath)[3::4]:
        print('Treating', fileName)
        y, sr = librosa.load(path=os.path.join(dataPath, fileName), sr=sRate)
        D = numpy.abs(librosa.stft(y, n_fft=nFFT, win_length=winLength, hop_length=hopLength,
                                   window=scipy.signal.hamming, center=False)) ** 2
        S = librosa.feature.melspectrogram(S=D, n_mels=mBands)
        gram = librosa.power_to_db(S, ref=numpy.max)
        gram = numpy.transpose(gram, (1, 0))

        with open(os.path.join(savePath, fileName.replace('wav', 'csv')), 'w') as file:
            for indexX in range(len(gram)):
                for indexY in range(len(gram[indexX])):
                    if indexY != 0: file.write(',')
                    file.write(str(gram[indexX][indexY]))
                file.write('\n')

        # exit()
