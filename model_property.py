from numpy import product
from models.layers.Basic_Layers import Static_Layernorm 

def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])

def static_layernorm_lip_const(model):
    layernorms = [i for i in model.modules() if type(i) == Static_Layernorm]
    lip_const_list = [float(i.std) for i in layernorms]
    return 1 / product(lip_const_list), lip_const_list

def static_layernorm_lip_loss(model):
    layernorms = [i for i in model.modules() if type(i) == Static_Layernorm]
    ret = 1
    for i in layernorms:
        ret = ret + 1 / i.last_std
    return ret