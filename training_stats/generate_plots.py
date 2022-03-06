from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os.path as path

plt.style.use('seaborn')

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
    data_0 = loadCSVData("./adam_0.7.csv")
    data_1 = loadCSVData("./adam_0.8.csv")
    data_2 = loadCSVData("./adam_0.85.csv")
    data_3 = loadCSVData("./adam_0.9.csv")
    data_4 = loadCSVData("./adam_0.97.csv")
    
    f1, ax1 = plt.subplots()
    ax1.plot(data_0[:, 0], data_0[:, 1], label=r'$\beta_1 = 0.7$')
    ax1.plot(data_1[:, 0], data_1[:, 1], label=r'$\beta_1 = 0.8$')
    ax1.plot(data_2[:, 0], data_2[:, 1], label=r'$\beta_1 = 0.85$')
    ax1.plot(data_3[:, 0], data_3[:, 1], label=r'$\beta_1 = 0.9$')
    ax1.plot(data_4[:, 0], data_4[:, 1], label=r'$\beta_1 = 0.97$')
    ax1.legend()
    ax1.set_ylabel('Training Loss')
    ax1.set_xlabel('Batch')
    ax1.set_title(r'Training Loss using Adam Optimizer While Varying $\beta_1$')
    plt.savefig('./optimizer_results.svg', format='svg')
