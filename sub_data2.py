# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 22:36:39 2021

@author: BMCL22
"""

import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat
import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import argparse
import glob
import math
import ntpath
import os
import shutil

from wfdb.processing.peaks import find_peaks
from datetime import datetime
#import neurokit2 as nk
import numpy as np
import scipy as sc

import math
# =============================================================================
# record = wfdb.rdrecord("mit-bih-arrhythmia-database-1.0.0/234")
# annotation = wfdb.rdann("mit-bih-arrhythmia-database-1.0.0/234", extension="atr")
# =============================================================================

# =============================================================================
# record = wfdb.rdrecord("Supraventricular_Arrhythmia_Database/845")
# annotation = wfdb.rdann("Supraventricular_Arrhythmia_Database/845", extension="atr")
# =============================================================================


class Segment:
    def __init__(self, data, peak, label):
        self.data = data
        self.peak = peak
        self.label = label

    def WindowMethod(self):
        '''
        segment in specipic window.
        '''
        signals = []
        peaks = []
        labels = []
        peaks.append(self.peak[1])
        for i in range(2, len(self.peak) - 1):  # fixed window size
            # MIT: 170 MIT_Normal:100
            diff1 = abs(self.peak[i] - 100)  # -100 for 250 / -70 for 170 / -60 for 100
            diff2 = abs(self.peak[i] + 150)  # 150 for 250 / 100 for 170 / 40 for 100
            signal = self.data[diff1:diff2]
            signals.append(signal)
            peaks.append(self.peak[i])
            labels.append(self.label.iloc[i])
        
        peaks.append(self.peak[len(self.peak)-1])
        self.peak = peaks
        self.label = labels
        return signals, peaks, labels

def peak_change(peaks,signal):
    real_peak = []

    for p in peaks:
        peak = 0
        temp = np.argmax(signal[p-5:p+5])
        if temp > 4:
            peak = p + (temp - 4)
        if temp == 4:
            peak = p
        if temp < 4:
            peak = p - (4 - temp)

        real_peak.append(peak)

    return real_peak

psg_fnames = glob.glob(os.path.join("./mit-database", "*.dat"))
#psg_fnames = glob.glob(os.path.join("./incart-database/files", "*.dat"))
#psg_fnames = glob.glob(os.path.join("./mit_raw_filtering", "*.dat"))
psg_fnames.sort()
psg_fnames = np.asarray(psg_fnames)

os.makedirs("./mit_mat_segment",exist_ok=True)
os.makedirs("./mit_mat_Rpeak",exist_ok=True)

for i in range(len(psg_fnames)):
    subject = psg_fnames[i]
    record = wfdb.rdrecord(subject[:-4])
    annotation = wfdb.rdann(subject[:-4], extension="atr")
    subject_name = subject[15:-4]
    #signal = record.p_signal[:, 0]
    mat = io.loadmat('./mit_mat_raw_filtering/'+ subject_name + ".mat")
    signal = mat.get('x')
    signal = np.reshape(signal,(signal.shape[1],))
    peaks = annotation.sample
    #times = [i for i in range(len(signal))]

    #peaks = peak_change(peaks[1:-1],signal)
    labels = pd.DataFrame(annotation.symbol)
    labels_ = pd.Series(annotation.symbol)
    
    method = Segment(data=signal, peak=peaks, label=labels)
    
    seg_signal_w, seg_peak_w, seg_label_w = method.WindowMethod()     
    
    seg_signal_w = pd.DataFrame(seg_signal_w)
    signal = np.array(seg_signal_w)
    peak = np.array(seg_peak_w)
    savemat("./mit_mat_segment/"+subject_name+".mat",{'x':signal})
    savemat("./mit_mat_Rpeak/"+subject_name+"_Rpeak.mat",{'p':peak})
    print(subject_name + "_end")

# =============================================================================
# seg_signal_w = pd.DataFrame(seg_signal_w)
# seg_signal_w.to_csv('Supraventricular_signal_825.csv', index=False)
# =============================================================================
        
# =============================================================================
# np.savetxt('MIT_BIH_Normal_16265.csv', seg_signal_w, fmt ='% s')
# =============================================================================

# =============================================================================
# for i in range(len(seg_label_w[:])):
#     
#     if seg_label_w[i] == "N" or seg_label_w[i] == "L" or seg_label_w[i] == "R" or seg_label_w[i] == "e" or seg_label_w[i] == "j" :
#         seg_label_w[i] = 0
#     
#     else :
#         seg_label_w[i] = 1
# =============================================================================

# =============================================================================
# def plot(data, peaks, time):
#     ihline_list = np.arange(-2, 1.5, 0.1)
# # =============================================================================
# #      for i in peaks:
# #          plt.axvline(i, linestyle=':', linewidth='0.7', color='gray')
# #      for i in ihline_list:
# #          plt.axhline(i, linestyle=':', linewidth='0.7', color='gray')
# # =============================================================================
#     plt.plot(time, data)
#     plt.plot(peaks, data[peaks], 'rv')
#     plt.xlim(0, 10000)
#     plt.ylim(-2, 2)
#     plt.show()
# =============================================================================
