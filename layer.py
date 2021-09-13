# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 22:40:14 2021

@author: Kaima
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily
from distribution import LogAl


class temporal_prob(nn.Module):
    # tau prediction
    def __init__(self, n_component = 32, in_features = 20, a = 1):
        super(temporal_prob, self).__init__() 
        self.n_component = n_component
        self.cat_linear = nn.Linear(in_features, n_component)
        self.mean_linear = nn.Linear(in_features, n_component)
        self.var_linear = nn.Linear(in_features, n_component)
        self.k_linear = nn.Linear(in_features, n_component)
        self.n_component = n_component
        self.a = a
        
    def prob_para(self, h):
        cat = self.cat_linear(h)
        mean = self.mean_linear(h)
        var = self.var_linear(h)
        k = self.k_linear(h)
        return torch.cat([cat, mean, var, k], dim = -1)
        
    def log_prob(self, h, tau):

        cat = torch.log_softmax(self.cat_linear(h), dim=-1)
        mean = self.mean_linear(h)
        var = torch.exp(self.var_linear(h))
        k = torch.exp(self.k_linear(h))        
        
        category = D.Categorical(logits=cat)
        t_dist = D.Independent(LogAl(mean, var, k), 0)
        gmm = MixtureSameFamily(category, t_dist)

        return gmm.log_prob(tau + 1e-5) - torch.log(self.a)
    
    def sample(self, h):
       
        cat = torch.log_softmax(self.cat_linear(h), dim=-1)
        mean = self.mean_linear(h)
        var = torch.exp(self.var_linear(h))
        k = torch.exp(self.k_linear(h))        
        
        category = D.Categorical(logits=cat)
        t_dist = D.Independent(LogAl(mean, var, k), 0)
        gmm = MixtureSameFamily(category, t_dist)
                
        return gmm.sample()   
    
class learnEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(learnEmbedding, self).__init__()
        self.factor = nn.parameter.Parameter(torch.randn(1,)).to('cuda')
        self.d_model = d_model
    
    def forward(self, x):
        div = torch.arange(0, self.d_model, 2).to('cuda')
        div_term = torch.exp(div * self.factor)
        v1 = torch.sin(torch.einsum('bt, f->btf', x, div_term))
        v2 = torch.cos(torch.einsum('bt, f->btf', x, div_term))
        return torch.cat([v1, v2], -1)
    
    
class ATT(nn.Module):
    def __init__(self,c_in,c_out, d = 16, device = 'cuda'):
        super(ATT,self).__init__()
        self.d = d
        self.qm = nn.Linear(in_features = c_in, out_features = d, bias = False)
        self.km = nn.Linear(in_features = c_in, out_features = d, bias = False)
        self.vm = nn.Linear(in_features = c_in, out_features = c_out, bias = False)
        self.device = device
        
    def forward(self,x):
        [_, T, _] = x.shape
       # mask = log_mask(T, int(3 *self.kernel)).view(1, T, T).to(device)
        mask = torch.tril(torch.ones(T, T)).view(1, T, T).to(self.device)
        query = self.qm(x)
        key = self.km(x)
        attention = torch.einsum('btf,bpf->btp', query, key)
        attention /= (self.d ** 0.5)
        attention = attention * mask + -1e9 * (1 - mask)
        attention = F.softmax(attention, dim=-1) 
        value = self.vm(x)
        out = torch.einsum('btp,bpf->btf', attention, value) # (avoid future leakage)
        return out # log mask in the future
    
