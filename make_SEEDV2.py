# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BIOT code base
# https://github.com/ycq091044/BIOT
# --------------------------------------------------------
import os
import pickle
import scipy
from multiprocessing import Pool
import numpy as np
import mne
from scipy import signal
labels = [1, 3, 4, 2, 7, 7, 2, 4, 3, 1, 1, 3, 4, 2, 7, 7, 2, 4, 3, 1, 7, 2, 5, 3, 6, 6, 3, 5, 2, 7, 7, 2, 5, 3, 6, 6, 3, 5, 2, 7, 1, 6, 4, 5, 7, 7, 5, 4, 6, 1, 1, 6, 4, 5, 7, 7, 5, 4, 6, 1, 4, 2, 5, 6, 1, 1, 6, 5, 2, 4, 4, 2, 5, 6, 1, 1, 6, 5, 2, 4]
chOrder_standard = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']
base = "EEG_preprocessed/raw"
dump_folder = "EEG_preprocessed/processed_4"
patient_path = os.listdir(base)
for mat_path in patient_path:
    data = scipy.io.loadmat(os.path.join(base, mat_path))  # 读取mat文件
    last_80_items = list(data.items())[-80:]  # 转为列表并切片

    for key, value in last_80_items:
        trial = None
        label = labels[int(key)-1]
        for channel in value:
            data = channel
            b, a = signal.butter(4, [0.1, 75], 'bandpass', fs=200)
            filtedData = signal.filtfilt(b, a, data)
            bn, an = signal.iirnotch(w0=50, Q=30, fs=200)
            notchData = signal.filtfilt(bn, an, filtedData)
            num = round(len(notchData)/5)
            resampleData = signal.resample(notchData, num)
            resampleData = np.expand_dims(resampleData, 1)
            resampleData = resampleData.transpose(1, 0)
            if trial is None:
                trial = resampleData
            else:
                trial = np.concatenate((trial, resampleData), axis=0)   # channel,time

        trial_cut = trial[:, -2000:]
        dump_path = os.path.join(
            dump_folder, mat_path.split(".")[0] + "_" + str(key) + ".pkl"
        )
        pickle.dump(
            {"X": trial[:, -2000:], "y": label},
            open(dump_path, "wb"),
        )



# 34 33 32 31
print("Done")

