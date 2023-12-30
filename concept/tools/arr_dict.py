import torch


def apply(d, f):
    if isinstance(d, torch.Tensor):
        return f(d)
    elif isinstance(d, dict):
        return {k: apply(d[k], f) for k in d}
    elif isinstance(d, list):
        return [apply(x, f) for x in d]
    elif isinstance(d, tuple):
        return tuple(apply(x, f) for x in d)
    else:
        raise NotImplementedError(f"apply() not implemented for {type(d)}")


def to(d, device):
    return apply(d, lambda x: x.to(device))



    
