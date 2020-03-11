import h5py

if __name__ == '__main__':
    file = h5py.File(name=r'D:\PythonProjects_Data\CMU_MOSEI\CMU_MOSEI_LabelsEmotions.csd', mode='r')
    for sample in file['Emotion Labels/data/--qXJuDtHPw/features']:
        print(sample)
