import numpy as np

def scale_z(params, mean, std):
    return (params - mean) / std


def unscale_z(params, mean, std):
    return params * std + mean