# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 22:34:02 2021

@author: Kaima
"""

import pandas as pd
import numpy as np

frame = []

for i in range(25):
    if i+1<10:
        trajectory_csv = pd.read_csv('Hangzhou-mobility-data-set/record_2019-01-0'+ str(i+1) + '.csv')
    else:
        trajectory_csv = pd.read_csv('Hangzhou-mobility-data-set/record_2019-01-'+ str(i+1) + '.csv')
    frame.append(trajectory_csv[trajectory_csv['payType'] != 3])
    
trajectory_csv = pd.concat(frame)

a = trajectory_csv['userID'].unique()

N = 0
u = 0
p_traj = []
all_traj = []
while N < 20000:
    u += 1
    df0 = trajectory_csv[trajectory_csv['userID'] == a[u]]
    if len(df0) > 20:
        df0['shijian'] = pd.to_datetime(df0['time'], format = "%Y-%m-%d %H:%M:%S")
        df0.sort_values(by ='shijian')
        for i in range(len(df0)//2):
            if df0.iloc[2*i]['status'] == 1 and df0.iloc[2*i + 1]['status'] == 0:
                traj_temp = np.zeros(6,)
                traj_temp[0] = df0.iloc[2*i]['stationID']
                traj_temp[1] = df0.iloc[2*i]['shijian'].hour
                traj_temp[2] = df0.iloc[2*i+1]['stationID']
                traj_temp[4] = df0.iloc[2*i]['shijian'].dayofweek
                
                if i == 0 and df0.iloc[2*i]['status'] == 1:
                    traj_temp[3] = 0
                else:
                    traj_temp[3] = (df0.iloc[2*i]['shijian'] - df0.iloc[2*i - 2]['shijian'])/ np.timedelta64(1, 's')
                    
                if df0.iloc[2*i]['shijian'].day >= 20:
                    traj_temp[5] = 1
                
                p_traj.append(traj_temp)
        if len(p_traj) > 5:
            log = 'Saved {}, length: {}'
            print(log.format(a[u],len(p_traj)))
            all_traj.append(np.array(p_traj))        
            N += 1
            p_traj = []
            
            
np.savez("trajectoryH20000.npz", *all_traj)

data = np.load("trajectoryH20000.npz")
all_traj = []

p = data.files

for i in range(len(p)):
    temp = data[p[i]][data[p[i]][:, 5] == 0, :]
    if temp.shape[0] > 3:
        all_traj.append(temp)
        
np.savez("rnn_trainingH.npz", *all_traj)

