# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 22:32:18 2021

@author: Kaima
"""
import torch
from math import radians, cos, sin, asin, sqrt

PAD_token = 0

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def maskNLLLoss(inp, target, mask = None):
    loss = 0 
    if not mask == None:    
        number = mask.sum()
    for i in range(len(inp)):
        ce = -target[i] * torch.log(inp[i] + 1e-9)
        if mask == None:
            loss += torch.sum(ce)
        else:
            loss += torch.sum(ce[mask[i]])
    if mask == None:
        return loss
    return loss

def maskAccuracy(inp, target, mask):
    winners = inp.argmax(dim=2)
    winners_label = target.argmax(dim=2)
    corrects = (winners == winners_label)
    accuracy = corrects[mask].sum().float()
    return accuracy

def temporal_distance(inp, target, mask):
    winners = inp.argmax(dim=2)
    winners_label = target.argmax(dim=2)
    distance = torch.abs(winners - winners_label)
    distance = distance[mask].sum().float()
    return distance

def haversine(lon1, lat1, lon2, lat2): 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r #km

class Log_StandardScaler():
    """
    Standard the input
    """
    def __init__(self, std):
        self.std = std

    def transform(self, data):
        return data / self.std

    def inverse_transform(self, data):
        return data * self.std