# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 22:58:06 2021

@author: Kaima
"""
import numpy as np
from utils import Log_StandardScaler, binaryMatrix, maskNLLLoss
import torch
from model import MIX_OD_ATT
import random
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

device = 'cuda'
PAD_token = 0
data = np.load("rnn_trainingH.npz")
p = data.files
rand = np.random.RandomState(0)                      #seed can be varied
b = rand.choice(p,int(0.1 * len(p)),replace=False)   #ratio can be other value
b = set(b)       
a = set(p) - b
a = list(a)
b = list(b)
t_sum = 0
num = 0
t_square = 0
for i in range(len(a)):
    t_sum += np.sum(data[a[i]][:, 3]/3600)
    t_square += np.sum(data[a[i]][:, 3]/3600 * data[a[i]][:, 3]/3600)
    num += len(data[a[i]][:, 3])
t_scaler = Log_StandardScaler(np.sqrt(t_square/num - (t_sum/num)**2))
ba = torch.tensor(np.sqrt(t_square/num - (t_sum/num)**2)).to(device)
ATT_ODTAU = MIX_OD_ATT(81, 81, 24, 7, 64, 64, 100, 8, 1e-3, 16, 2, ba)
ATT_ODTAU = ATT_ODTAU.to(device)
batch_size = 32

for e in range(140):
    loss_sum = 0
    random.shuffle(a)
    for i in range(len(a)//batch_size - batch_size):
        batch_ti = []
        batch_oi = []
        batch_di = []
        batch_hi = []
        batch_wi = []
        
        batch_to = []
        batch_oo = []
        batch_do = []
        #batch_tof = []
        for j in range(batch_size):  
            #fake_to = fake_tau(data[a[i*batch_size + j]][:-1, 1])
            #batch_tof.append(torch.Tensor(t_scaler.transform(fake_to)).to(device))
            
            batch_ti.append(torch.Tensor(t_scaler.transform(data[a[i*batch_size + j]][:-1, 3]/3600)).to(device))
            batch_oi.append(torch.LongTensor(data[a[i*batch_size + j]][:-1, 0]).to(device))
            batch_di.append(torch.LongTensor(data[a[i*batch_size + j]][:-1, 2]).to(device))    
            batch_hi.append(torch.LongTensor(data[a[i*batch_size + j]][:-1, 1]).to(device))
            batch_wi.append(torch.LongTensor(data[a[i*batch_size + j]][:-1, 4]).to(device))  
            batch_to.append(torch.Tensor(t_scaler.transform(data[a[i*batch_size + j]][1:, 3]/3600)).to(device))
            batch_oo.append(torch.LongTensor(data[a[i*batch_size + j]][1:, 0]).to(device))
            batch_do.append(torch.LongTensor(data[a[i*batch_size + j]][1:, 2]).to(device))
                
        batch_ti = pad_sequence(batch_ti, batch_first = True)
        batch_oi = pad_sequence(batch_oi, batch_first = True)
        batch_di = pad_sequence(batch_di, batch_first = True)
        batch_hi = pad_sequence(batch_hi, batch_first = True)
        batch_wi = pad_sequence(batch_wi, batch_first = True)
        batch_to = pad_sequence(batch_to, batch_first = True)
        batch_oo = pad_sequence(batch_oo, batch_first = True)
        batch_do = pad_sequence(batch_do, batch_first = True)
        
        batch_ti = batch_ti.unsqueeze(-1)
        batch_to = batch_to.unsqueeze(-1)
        #batch_tof = batch_tof.unsqueeze(-1)
        
        m0 = binaryMatrix(batch_oo, value=PAD_token)
        m1 = binaryMatrix(batch_do, value=PAD_token)
        for i in range(len(m0)):
            m0[i] = m0[i] or m1[i]
            
        mask = torch.BoolTensor(m0).to(device)
        
        o_label = nn.functional.one_hot(batch_oo, num_classes=ATT_ODTAU.o_class).float()
        d_label = nn.functional.one_hot(batch_do, num_classes=ATT_ODTAU.d_class).float()

        hid, o_out, d_out = ATT_ODTAU(batch_ti, batch_oi, batch_di, batch_hi, batch_wi)
        mark_loss = maskNLLLoss(d_out, d_label, mask) + maskNLLLoss(o_out, o_label, mask) 
        tau_prob = ATT_ODTAU.fstar(batch_to.squeeze(-1), hid)
        tau_loss = -torch.sum(tau_prob[mask])
        loss = mark_loss + tau_loss #+ 0.1*fake_tau_loss
        
        loss.backward()
        nn.utils.clip_grad_norm_(ATT_ODTAU.parameters(), 3)
        ATT_ODTAU.optimizer.step()
        ATT_ODTAU.optimizer.zero_grad()
        
        loss_sum += loss.item()
    val_error = 0    
    for i in range(len(b)):
        batch_ti = (torch.Tensor(t_scaler.transform(data[b[i]][:-1, 3]/3600)).to(device)).unsqueeze(0)
        batch_oi = (torch.LongTensor(data[b[i]][:-1, 0]).to(device)).unsqueeze(0)
        batch_di = (torch.LongTensor(data[b[i]][:-1, 2]).to(device)).unsqueeze(0)     
        batch_hi = (torch.LongTensor(data[b[i]][:-1, 1]).to(device)).unsqueeze(0)
        batch_wi = (torch.LongTensor(data[b[i]][:-1, 4]).to(device)).unsqueeze(0)    
        
        batch_to = (torch.Tensor(t_scaler.transform(data[b[i]][1:, 3]/3600)).to(device)).unsqueeze(0)
        batch_oo = (torch.LongTensor(data[b[i]][1:, 0]).to(device)).unsqueeze(0)
        batch_do = (torch.LongTensor(data[b[i]][1:, 2]).to(device)).unsqueeze(0)
        
        batch_ti = batch_ti.unsqueeze(-1)
        batch_to = batch_to.unsqueeze(-1)
        o_label = nn.functional.one_hot(batch_oo, num_classes=ATT_ODTAU.o_class).float()
        d_label = nn.functional.one_hot(batch_do, num_classes=ATT_ODTAU.d_class).float()
        with torch.no_grad():
            hid, o_out, d_out = ATT_ODTAU(batch_ti, batch_oi, batch_di, batch_hi, batch_wi)
            mark_loss = maskNLLLoss(d_out, d_label) + maskNLLLoss(o_out, o_label)
            tau_prob = ATT_ODTAU.fstar(batch_to.squeeze(-1), hid)
            tau_loss = -torch.sum(tau_prob)
            loss = mark_loss + tau_loss
        val_error += loss.item()
        
    log = 'Episode {}, Training_loss: {} Evaluation_loss: {}'
    print(log.format(e,loss_sum, val_error))



