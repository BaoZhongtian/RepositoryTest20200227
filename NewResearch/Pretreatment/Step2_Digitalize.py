import numpy

if __name__ == '__main__':
    movieData = numpy.genfromtxt(fname='MovieData.csv', dtype=str, delimiter='::')

    dictionary = {}
    for sample in movieData:
        # print(sample)
        sample = sample[-1].split('|')
        for subsample in sample:
            if subsample not in dictionary.keys(): dictionary[subsample] = len(dictionary.keys())
    print(dictionary)

    with open('MovieData_Digital.csv', 'w') as file:
        for sample in movieData:
            # print(sample)
            file.write(sample[0] + '::' + sample[2])

            sample = sample[-1].split('|')
            for index, subsample in enumerate(dictionary.keys()):
                if subsample in sample:
                    file.write('::1')
                else:
                    file.write('::0')
            file.write('\n')
    #########################################
    # data = numpy.genfromtxt(fname='MovieData_Digital.csv', dtype=int, delimiter='::')
    # print(numpy.shape(data))
    #########################################
    data = numpy.genfromtxt(fname='UserData.csv', dtype=str, delimiter='::')
    dictionary = {}
    for sample in data:
        if sample[2] not in dictionary.keys(): dictionary[sample[2]] = len(dictionary.keys())

    with open('UserData_Digital.csv', 'w') as file:
        for sample in data:
            print(sample)
            file.write(sample[0] + '::')
            if sample[1] == 'F':
                file.write('0::')
            else:
                file.write('1::')
            file.write(str(dictionary[sample[2]]) + '::')
            file.write(sample[3] + '\n')
