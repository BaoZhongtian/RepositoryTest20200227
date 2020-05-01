import numpy

if __name__ == '__main__':
    userData = numpy.genfromtxt(fname='UserData_Digital.csv', dtype=str, delimiter='::')
    userDictionary = {}
    for sample in userData:
        userDictionary[int(sample[0])] = '%s::%s::%s' % (sample[1], sample[2], sample[3])

    print('User Load Completed')

    movieData = numpy.genfromtxt(fname='MovieData_Digital.csv', dtype=str, delimiter='::')
    movieDictionary = {}
    for sample in movieData:
        movieDictionary[int(sample[0])] = sample[1]
        for index in range(2, len(sample)):
            movieDictionary[int(sample[0])] += '::' + sample[index]

    print('Movie Load Completed')

    recordData = numpy.genfromtxt(fname='RecordData.csv', dtype=int, delimiter='::')
    with open('RecordData_Supplement.csv', 'w') as file:
        for sample in recordData:
            if sample[0] not in userDictionary.keys() or sample[1] not in movieDictionary.keys(): continue
            file.write(str(sample[0]) + '::' + userDictionary[sample[0]] + '::')
            file.write(str(sample[1]) + '::' + movieDictionary[sample[1]] + '::')
            file.write(str(sample[2]) + '\n')
