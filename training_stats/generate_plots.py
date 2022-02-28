from matplotlib import pyplot as plt
import numpy as np
import os.path as path

# read the log data
def loadCSVData(filename):
    input_file = path.join(path.dirname(__file__), filename)
    data_list = list()
    with open(input_file, 'r') as fd:
        lines = [line.strip('\n') for line in fd.readlines()]
        for i in range(1,len(lines)):
            data_list.append(list(map(float, lines[i].split(','))))
    return np.array(data_list)

if __name__ == "__main__":
    data = loadCSVData("batch_log.csv")
    plt.plot(data[:,0], data[:,1])
    plt.show()
    plt.plot(data[:,0], data[:, 2])
    plt.show()
        