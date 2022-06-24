# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:59:02 2021

@author: BCML15
"""
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
import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse

from datetime import datetime

import numpy as np

#psg_fnames = glob.glob(os.path.join("./mit-database", "*.dat"))
#psg_fnames = glob.glob(os.path.join("./european-database", "*.dat"))
#psg_fnames = glob.glob(os.path.join("./cu-database", "*.dat"))
psg_fnames = glob.glob(os.path.join("./incart-database/files", "*.dat"))
psg_fnames.sort()
psg_fnames = np.asarray(psg_fnames)

#os.makedirs("./mit_mat_Rpeak",exist_ok=True)
#os.makedirs("./mit_mat_raw",exist_ok=True)
os.makedirs("./incart_mat_raw",exist_ok=True)
for i in range(len(psg_fnames)):
    subject = psg_fnames[i]
    record = wfdb.rdrecord(subject[:-4])
    annotation = wfdb.rdann(subject[:-4], extension="atr")

    # =============================================================================
    # record = wfdb.rdrecord("Supraventricular_Arrhythmia_Database/845")
    # annotation = wfdb.rdann("Supraventricular_Arrhythmia_Database/845", extension="atr")
    # =============================================================================
    # record = wfdb.rdrecord("MIT_BIH_Normal_Database/19830")
    # annotation = wfdb.rdann("MIT_BIH_Normal_Database/19830", extension="atr")

    signal = record.p_signal[:, 0]
    peaks = annotation.sample
    times = [i for i in range(len(signal))]
    labels = pd.DataFrame(annotation.symbol)
    labels_ = pd.Series(annotation.symbol)
    
    print(labels_.value_counts())
             
    #savemat("./mit_mat/"+subject[14:-4]+".mat",{'x':seg_signal_w[2:]})
    # savemat("./mit_mat/"+subject[14:-4]+".mat",{'x':seg_signal_w})
    # savemat("./mit_mat_Rpeak/"+subject[14:-4]+"_Rpeak.mat",{'p':seg_peak_w})
    #savemat("./mit_mat_raw/"+subject[14:-4]+".mat",{'x':signal})
    #savemat("./edb_mat_raw/" + subject[20:-4] + ".mat", {'x': signal})
    savemat("./incart_mat_raw/" + subject[24:-4] + ".mat", {'x': signal})
    #seg_signal_w = pd.DataFrame(seg_signal_w)
    #seg_signal_w.to_csv('MIT_BIH_Normal_19830.csv', index=False)
    