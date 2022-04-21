from turtle import color
from cv2 import dft
from matplotlib import pyplot as plt
import numpy as np
import os.path as path
import pandas as pd
import json

from sklearn.datasets import load_boston

plt.style.use('seaborn')

CLASS_LABELS = ["BLA", "LYT", "NGB", "NGS"]
RECALL_LABELS = ["val_bla-recall", "val_lyt-recall", "val_ngb-recall", "val_ngs-recall"]
PRECISION_LABELS = ["val_bla-precision", "val_lyt-precision", "val_ngb-precision", "val_ngs-precision"]

# read the log data
def loadData(filename):
    input_file = path.join(path.dirname(__file__), filename)
    lines = list()
    with open(input_file, 'r') as fd:
        lines = [pd.Series(json.loads(str.replace(line.strip('\n'), "\'", "\""))) for line in fd.readlines()]
    return pd.DataFrame(lines)

def GenLossNAccraucyPlot(filepath, dataframe, plot_title):
    fig, ax1 = plt.subplots(2,1)
    fig.suptitle(plot_title, fontsize= 10)
    fig.supxlabel('Epoch', fontsize= 10)
    ax1[0].plot(dataframe.index, dataframe.get('val_loss'), '-')
    ax1[0].set_title('Loss', fontsize= 9)
    ax1[1].plot(dataframe.index, dataframe.get('val_acc'), '-')
    ax1[1].set_title('Categorical Accuracy', fontsize= 9)
    plt.savefig(filepath, dpi=300)
    plt.show()

def GenRecallNPrecisionPlot(filepath, dataframe, plot_title):
    fig, ax1 = plt.subplots(2,1)
    fig.suptitle(plot_title, fontsize= 10)
    fig.supxlabel('Epoch', fontsize= 10)
    ax1[0].plot(dataframe.index, dataframe.get(RECALL_LABELS), '-')
    ax1[0].set_title('Recall', fontsize= 9)
    ax1[1].plot(dataframe.index, dataframe.get(PRECISION_LABELS), '-')
    ax1[1].set_title('Precision', fontsize= 9)
    fig.legend(CLASS_LABELS, fontsize=9)
    plt.savefig(filepath, dpi=300)
    plt.show()

if __name__ == "__main__":

    recall_filepath = path.join(path.abspath("./MFinal-Recall.png"))
    acc_filepath = path.join(path.abspath("./MFinal-Precision.png"))
    final_res = loadData(f'./final-model-results.csv')

    GenLossNAccraucyPlot(acc_filepath, final_res, "MV3 Learning Rate = 0.0001 - Accuracy & Loss")
    GenRecallNPrecisionPlot(recall_filepath, final_res, "MV3 Learning Rate = 0.0001 - Recall & Precision")

    # plt.xlabel('Epoch', fontsize= 10)
    # plt.ylabel('Loss', fontsize= 10)
    # plt.plot( [i for i in range(1,21,1)], mv3_lr2_gauss.get('val_loss')[0:20], '--', color='green')
    # plt.plot( [i for i in range(1,21,1)], mv3_lr2.get('val_loss')[0:20], '--', color='orange')
    # plt.plot( [i for i in range(1,21,1)], mv3_lr1.get('val_loss')[0:20], '--', color='red')
    # plt.plot( [i for i in range(1,21,1)], mv3_std.get('val_loss')[0:20], '--', color='black')
    # plt.plot( [i for i in range(1,21,1)], mv3_lr2_gauss.get('loss')[0:20], '-', color='green')
    # plt.plot( [i for i in range(1,21,1)], mv3_lr2.get('loss')[0:20], '-', color='orange')
    # plt.plot( [i for i in range(1,21,1)], mv3_lr1.get('loss')[0:20], '-', color='red')
    # plt.plot( [i for i in range(1,21,1)], mv3_std.get('loss')[0:20], '-', color='black')
    # plt.title('MV3 - Validation vs. Training Loss Convergence', fontsize= 10)
    # plt.legend(['LR = 0.0001 w/ Gaussian Noise', 'LR = 0.0001', 'LR = 0.001', 'LR = 0.003'])
    # plt.xlim([0,21])
    # plt.ylim([0,2])
    # plt.show()
    # print(np.mean(mv3_std.get('val_loss')[7:20]))
    # print(np.mean(mv3_lr1.get('val_loss')[7:20]))
    # print(np.mean(mv3_lr2.get('val_loss')[7:20]))
    # print(np.mean(mv3_lr2_gauss.get('val_loss')[7:20]))
    # print(np.mean(mv3_lr2_gauss_l2.get('val_categorical_crossentropy')[7:20]))
    


    