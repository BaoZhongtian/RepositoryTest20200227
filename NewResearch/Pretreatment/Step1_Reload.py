import numpy

if __name__ == '__main__':
    with open(r'D:\PythonProjects\RepositoryTest20200227\ml-1m\movies.dat', 'r', errors='ignore') as file:
        data = file.readlines()
    with open('MovieData.csv', 'w') as file:
        for sample in data:
            sample = sample.split('::')

            file.write(sample[0] + '::')
            file.write(sample[1][0:-6] + '::')
            file.write(sample[1][-5:-1] + '::')
            file.write(sample[2])
            # exit()

    ######################################

    # with open(r'D:\PythonProjects\RepositoryTest20200227\ml-1m\users.dat', 'r', errors='ignore') as file:
    #     data = file.readlines()
    # with open('UserData.csv', 'w') as file:
    #     for sample in data:
    #         file.write(sample)
    #
    # #####################################
    #
    # with open(r'D:\PythonProjects\RepositoryTest20200227\ml-1m\ratings.dat', 'r', errors='ignore') as file:
    #     data = file.readlines()
    # with open('RecordData.csv', 'w') as file:
    #     for sample in data:
    #         file.write(sample)
