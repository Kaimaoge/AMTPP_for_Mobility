# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 22:48:11 2021

@author: Kaima
"""
import torch
import torch.nn as nn
from layer import learnEmbedding, ATT, temporal_prob
import torch.nn.functional as F

class MIX_OD_ATT(nn.Module):
    def __init__(self, o_class, d_class, t_class, w_class, s_emb, t_emb, hid_dim, head, lr, n_component, d_rank, t_a, device = 'cuda'):
        super(MIX_OD_ATT, self).__init__()
        '''
        o_class, d_class, t_class, w_class: number of origin, destination, hour of day and day of week
        s_emb, t_emb: hidden dimension of s and t embeddings
        hid_dim: attention dimension, head: number of head
        lr: learning rate, n_component: number of mixture distribution component
        d_rank: rank of OD matrix, t_a: the scaler for normalizing tau
        '''
        self.o_class = o_class
        self.d_class = d_class
        self.t_class = t_class
        self.w_class = w_class
        self.d_rank = d_rank
        #self.od_sparsity = od_sparsity
        self.head = head
#        self.POI = torch.FloatTensor(POI).to(device)
        self.origin_embedding = nn.Linear(in_features=self.o_class, out_features=s_emb)
        self.destin_embeeding = nn.Linear(in_features=self.d_class, out_features=s_emb)
        
        self.time_embedding = learnEmbedding(t_class, t_emb)
        self.week_embedding = learnEmbedding(w_class, t_emb)
        
        self.attn = nn.ModuleList()       

        for i in range(head):
            self.attn.append(ATT(2*s_emb + 2*t_emb + 1, hid_dim // head))
            
        self.att_out = nn.Linear(in_features=hid_dim // head * head, out_features=hid_dim, bias=False)
        
        self.o_linear = nn.Linear(in_features=hid_dim + 4*n_component, out_features=o_class, bias=True)
        self.d1 = nn.Linear(in_features=hid_dim + 4*n_component, out_features=o_class*d_rank, bias=True)
        self.d2 = nn.Linear(in_features=hid_dim + 4*n_component, out_features=d_class*d_rank, bias=True)
        
        self.t_mix = temporal_prob(n_component, hid_dim, t_a)
        self.att_act = nn.GELU()
        self.out_act = nn.Softmax(dim = 2)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        
    def fstar(self, u, hj):
        return self.t_mix.log_prob(hj, u)
    
    def next_time(self, hj):
        return self.t_mix.pred(hj)
        
    def forward(self, tau_input, o_input, d_input, t_input, w_input):
        #t_input = nn.functional.one_hot(t_input, num_classes=self.t_class).float()
        [batch, t_number, _] = tau_input.shape
        o_input = nn.functional.one_hot(o_input, num_classes=self.o_class).float()
        d_input = nn.functional.one_hot(d_input, num_classes=self.d_class).float()
        # o_poi = torch.matmul(o_input, self.POI)
        # d_poi = torch.matmul(d_input, self.POI)
        o_embedding = self.origin_embedding(o_input) # not smooth 
        d_embedding = self.destin_embeeding(d_input)
        
        t_embedding = self.time_embedding(t_input)
        w_embedding = self.week_embedding(w_input)
        
        sp_embedding = torch.cat((o_embedding, d_embedding), dim = -1)
        ti_embedding = torch.cat((t_embedding, w_embedding), dim = -1)
        
        att_input = torch.cat((tau_input, sp_embedding, ti_embedding), dim=-1)
        attn = []
        for i in range(self.head):
            attn.append(self.attn[i](att_input))
        attn = torch.cat(attn, dim = -1)
        
        hidden_state = self.att_act(self.att_out(attn))
        time_para = self.t_mix.prob_para(hidden_state)
        o_hidden = torch.cat([time_para, hidden_state], dim = -1)
        #t_out = self.out_act(self.t_linear(hidden_state))
        o_out = self.out_act(self.o_linear(o_hidden))
        d1_base = self.d1(o_hidden).reshape(batch, t_number, self.o_class, self.d_rank)
        d2_base = self.d2(o_hidden).reshape(batch, t_number, self.d_class, self.d_rank)   
        od_att = torch.einsum('btfr, bter->btfe', d1_base, d2_base) #= torch.einsum('btfr, bter->btfe', d1_base, d2_base)       
        own_mask = (torch.ones([batch, t_number, self.o_class, self.d_class]) - torch.diag(torch.ones(self.d_class))).to(self.device)
        od_att = od_att * own_mask + -1e9 *(1 - own_mask)
        od_att = F.softmax(od_att, dim = -1)
        d_out = torch.einsum('btd, btdc->btc', o_out, od_att)
        #d_out = self.out_act(self.d_linear(hidden_state))
        #tau_out = self.next_time(hidden_state)
        return hidden_state, o_out, d_out