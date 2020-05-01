import numpy

if __name__ == '__main__':
    data = numpy.genfromtxt(fname='readFile.txt', dtype=float, delimiter='\t')
    print(data)

    for indexX in range(numpy.shape(data)[0]):
        for indexY in range(numpy.shape(data)[1]):
            print((data[indexX][indexY] * 100 + 1 + numpy.random.rand()) / 100, end='\t')
        print()
