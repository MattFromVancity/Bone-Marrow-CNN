from imageio import save
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os.path as path
import pandas as pd
import json

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

if __name__ == "__main__":

    savePath = path.abspath("./Validation_Stats/")
    df_epoch1 = loadData(f'./Validation_Stats/ModelV2-10Epochs-20220319.csv')

    # fig, ax1 = plt.subplots(2,2)
    # fig.suptitle('Model V1 Training Recall and Precision', fontsize= 10)
    # fig.supxlabel('Batch', fontsize= 10)
    # ax1[0,0].plot(df_epoch1.index, df_epoch1.get(RECALL_LABELS))
    # ax1[0,0].set_title('Epoch 1', fontsize= 9)
    # ax1[0,0].set_ylabel('Recall', fontsize= 10)
    # ax1[0,1].plot(df_epoch2.index, df_epoch2.get(RECALL_LABELS))
    # ax1[0,1].set_title('Epoch 2', fontsize= 9)

    # ax1[1,0].plot(df_epoch1.index, df_epoch1.get(PRECISION_LABELS))
    # ax1[1,0].set_ylabel('Precision', fontsize= 10)

    # ax1[1,1].plot(df_epoch2.index, df_epoch2.get(PRECISION_LABELS))

    # fig.legend(CLASS_LABELS, frameon= False, fontsize= 9)
    # plt.savefig(path.join(savePath,"MV1-Train-Recall-Precision"), dpi=300)
    # plt.show()

    fig, ax1 = plt.subplots(2,1)
    fig.suptitle('Model V2 Validation Recall & Precision', fontsize= 10)
    fig.supxlabel('Epoch', fontsize= 10)
    #fig.supylabel('Accuracy', fontsize= 10)
    ax1[0].plot(df_epoch1.index, df_epoch1.get(RECALL_LABELS), '--')
    ax1[0].set_title('Recall', fontsize= 9)
    ax1[1].plot(df_epoch1.index, df_epoch1.get(PRECISION_LABELS), '--')
    ax1[1].set_title('Precision', fontsize= 9)
    fig.legend(CLASS_LABELS, fontsize=9)

    plt.savefig(path.join(savePath,"MV2-Validation-Recall+Precision-Results"), dpi=300)
    plt.show()
    