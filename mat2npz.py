# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:04:51 2021

@author: BCML15
"""
from scipy.io import savemat
import wfdb
import pandas as pd
import numpy as np


import argparse
import glob

import os
import shutil
import urllib.request, urllib.parse, urllib.error


import numpy as np


from scipy import io
from os import listdir,makedirs

# Label values
N = 0
S = 1
V = 2
F = 3
Q = 4
X = 5

class_dict = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

ann2label = {
    "R": 0,
    "L": 0,
    "N": 0,
    "e": 0,
    "j": 0,

    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,


    "V": 2,
    "E": 2,

    "F": 3,
    
    "f": 4,
    "Q": 4,
    "P": 4,
    "/": 4,
    "U": 4,
    
    "+": 5,
    "~": 5,
    "|": 5,
    '"': 5,
    "!": 5,
    "[": 5,
    "]": 5,
    "x": 5,
}
ann2label2 = {
    "e": 9,  # nope
    "j": 0,
    "R": 1,
    "L": 2,
    "N": 3,

    "A": 4,
    "a": 5,
    "J": 6,
    "S": 9,  # nope

    "V": 7,
    "E": 9,  # nope

    "F": 8,

    "f": 9,
    "Q": 9,
    "P": 9,
    "/": 9,
    "U": 9,

    "+": 9,
    "~": 9,
    "|": 9,
    '"': 9,
    "!": 9,
    "[": 9,
    "]": 9,
    "x": 9,
}


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./mit-database",
                        help="File path to the CSV or NPY file that contains walking data.")
args = parser.parse_args()

data_path = './mit_mat_TDfeature'
#make_data_path = './mit_4class_npz'
#make_data_path = './mit_TDfeature_npz'
make_data_path = './mit_raw_npz'
subject_list = [f for f in listdir(data_path)]
makedirs(make_data_path,exist_ok=True)
makedirs(make_data_path + '/test',exist_ok=True)
makedirs(make_data_path + '/training',exist_ok=True)
makedirs(make_data_path + '/paced beat',exist_ok=True)
psg_fnames = glob.glob(os.path.join(args.data_dir, "*.dat"))
psg_fnames.sort()
psg_fnames = np.asarray(psg_fnames)


for i in range(len(psg_fnames)):
    subject = psg_fnames[i]
    record = wfdb.rdrecord(subject[:-4])
    annotation = wfdb.rdann(subject[:-4], extension="atr")

    signal = record.p_signal[:, 0]
    peaks = annotation.sample
    times = [i for i in range(len(signal))]
    labels = pd.DataFrame(annotation.symbol)
    labels_ = pd.Series(annotation.symbol)
    
    print(labels_[3:-1].value_counts())
    
    #mat = io.loadmat('./mit_mat_TDfeature/'+subject[14:-3]+"mat")
    mat = io.loadmat('./mit_mat_segment/' + subject[14:-3] + "mat")
    x = mat.get('x') 
    #x = np.reshape(x,(int(x.shape[0]/6000),3000,1))     
    labels_np = labels_[2:-1].to_numpy()
    label_int =[]
    spectrum =[]
    test_sub_path = make_data_path+ '/test' +  subject[14:-3] + 'npz'
    paced_sub_path = make_data_path+ '/paced beat' +  subject[14:-3] + 'npz'
    train_sub_path = make_data_path+ '/training' +  subject[14:-3] + 'npz'
    
    for j in range(len(labels_np)):
        ann_str = "".join(labels_np[j])
        label = ann2label[ann_str]
        if label != 5 and label != 4 and label != 3:
            label_int.append(label)
            spectrum.append(x[j,:])

    y = np.array(label_int)
    x = np.array(spectrum)
    #x = np.reshape(x,(x.shape[0],1,x.shape[1]))
    save_dict = {
        "x": x, 
        "y": y,
    }
    
    n = int(subject[15:-4])
    if n == 100 or n ==103 or n ==105 or n ==111 or n ==113 or n ==117 or\
        n ==121 or n ==123 or n ==200 or n ==202 or n ==210 or n ==212 or \
            n ==213 or n ==214 or n ==219 or n ==221 or n ==222 or n ==228 or\
                n ==231 or n ==232 or n ==233 or n ==234:
                    
        np.savez(test_sub_path[:-4], **save_dict)
        
    elif n == 102 or n ==104 or n ==107 or n ==217:
        np.savez(paced_sub_path[:-4], **save_dict)
    else:
        np.savez(train_sub_path[:-4], **save_dict)
    print(subject + 'end\n')
