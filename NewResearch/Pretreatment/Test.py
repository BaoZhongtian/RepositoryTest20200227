import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    dictionary = {}
    data = numpy.genfromtxt(fname='MovieData.csv', dtype=str, delimiter='::')
    for sample in data:
        print(sample)
        if sample[-2] not in dictionary.keys():
            dictionary[sample[-2]] = 1
        else:
            dictionary[sample[-2]] += 1

    yearList = []
    for sample in dictionary.keys():
        yearList.append(sample)
    yearList = sorted(yearList)

    accumulateList = [0]
    for sample in yearList:
        accumulateList.append(accumulateList[-1] + dictionary[sample])
    print(accumulateList)
    print(yearList)
    plt.plot(accumulateList)
    plt.show()

    before, after = 0, 0
    for sample in dictionary.keys():
        if int(sample) <= 1998:
            before += dictionary[sample]
        else:
            after += dictionary[sample]
    print(before, after)
