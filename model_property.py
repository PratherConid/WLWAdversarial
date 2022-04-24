from numpy import product
import torch
from models.layers.Lip_Layer import Static_Layernorm 

def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])

def static_layernorm_lip_const(model):
    layernorms = [i for i in model.modules() if type(i) == Static_Layernorm]
    return 1 / product([i.std for i in layernorms])