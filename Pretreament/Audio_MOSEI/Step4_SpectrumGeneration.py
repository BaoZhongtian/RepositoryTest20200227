import os
import librosa
from scipy import signal
import numpy

if __name__ == '__main__':
    m_bands = 40
    loadpath = 'D:/PythonProjects_Data/CMU_MOSEI/AudioPart/Step3_SeparateFold/valid/'
    savepath = 'D:/PythonProjects_Data/CMU_MOSEI/AudioPart/Step4_SpectrumGeneration/valid/'
    os.makedirs(savepath)

    s_rate = 16000
    win_length = int(0.025 * s_rate)  # Window length 15ms, 25ms, 50ms, 100ms, 200ms
    hop_length = int(0.010 * s_rate)  # Window shift  10ms
    n_fft = win_length

    for filename in os.listdir(loadpath):
        print('Treating', filename)
        y, sr = librosa.load(path=os.path.join(loadpath, filename), sr=s_rate)
        try:
            D = numpy.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                       window=signal.hamming, center=False)) ** 2
        except:
            continue
        S = librosa.feature.melspectrogram(S=D, n_mels=m_bands)
        gram = librosa.power_to_db(S, ref=numpy.max)
        gram = numpy.transpose(gram, (1, 0))

        with open(os.path.join(savepath, filename.replace('.wav', '.csv')), 'w') as file:
            for indexX in range(len(gram)):
                for indexY in range(len(gram[indexX])):
                    if indexY != 0: file.write(',')
                    file.write(str(gram[indexX][indexY]))
                file.write('\n')
