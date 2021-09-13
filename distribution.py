# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 22:38:50 2021

@author: Kaima
"""

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.distributions.transforms import ExpTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from torch.distributions.distribution import Distribution


class AsymmetricLaplace(Distribution):
    arg_constraints = {
         "loc": constraints.real,
         "scale": constraints.positive,
         "asymmetry": constraints.positive,
     }
    support = constraints.real
    has_rsample = True
    
    def __init__(self, loc, scale, asymmetry, *, validate_args=None):
        self.loc, self.scale, self.asymmetry = broadcast_all(loc, scale, asymmetry)
        super().__init__(self.loc.shape, validate_args=validate_args)
        self.left_scale = self.scale * self.asymmetry
        self.right_scale = self.scale / self.asymmetry
    
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(AsymmetricLaplace, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.asymmetry = self.asymmetry.expand(batch_shape)
        super(AsymmetricLaplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new  
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = value - self.loc
        z = -z.abs() / torch.where(z < 0, self.left_scale, self.right_scale)
        return z - (self.left_scale + self.right_scale).log()
     
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u, v = self.loc.new_empty((2,) + shape).exponential_()
        return self.loc - self.left_scale * u + self.right_scale * v
    
    def mean(self):
        total_scale = self.left_scale + self.right_scale
        return self.loc + (self.right_scale ** 2 - self.left_scale ** 2) / total_scale
     
    def variance(self):
        left = self.left_scale
        right = self.right_scale
        total = left + right
        p = left / total
        q = right / total
        return p * left ** 2 + q * right ** 2 + p * q * total ** 2
    
class LogAl(TransformedDistribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'asymmetry': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, asymmetry, validate_args=None):
        base_dist = AsymmetricLaplace(loc, scale, asymmetry, validate_args=validate_args)
        super(LogAl, self).__init__(base_dist, ExpTransform(), validate_args=validate_args)
        
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogAl, _instance)
        return super(LogAl, self).expand(batch_shape, _instance=new)
    
    def loc(self):
        return self.base_dist.loc

    def scale(self):
        return self.base_dist.scale

    def mean(self):
        return (self.loc + self.scale.pow(2) / 2).exp()

    def variance(self):
        return (self.scale.pow(2).exp() - 1) * (2 * self.loc + self.scale.pow(2)).exp()
    
    def entropy(self):
        return self.base_dist.entropy() + self.loc