import torch
from collections import defaultdict

def _recursive_read(obj):
    # From https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/utils/comm.py
    values = defaultdict(list)
    sizes = defaultdict(list)
    if isinstance(obj, torch.Tensor):
        values[obj.dtype] += [obj.flatten()]
        sizes[obj.dtype] += [torch.tensor([obj.numel()], device=obj.device)]
    elif isinstance(obj, dict):
        for v in obj.values():
            child_values, child_sizes = _recursive_read(v)
            for k, v in child_values.items():
                values[k] += v
            for k, v in child_sizes.items():
                sizes[k] += v
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for v in obj:
            child_values, child_sizes = _recursive_read(v)
            for k, v in child_values.items():
                values[k] += v
            for k, v in child_sizes.items():
                sizes[k] += v
    else:
        raise ValueError("Unknown type `%s`" % type(obj))
    return values, sizes


def _recursive_write(obj, values, sizes=None):
    # From https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/utils/comm.py
    if isinstance(obj, torch.Tensor):
        if sizes is None:
            size = torch.tensor([obj.numel()], device=obj.device)
        else:
            s = sizes[obj.dtype]
            size, s = s.split([1, len(s) - 1])
            sizes[obj.dtype] = s
        v = values[obj.dtype]
        new_obj, v = v.split([size, v.shape[-1] - size], dim=-1)
        # compatible with reduce / stack / cat
        new_obj = new_obj.view(new_obj.shape[:-1] + (-1,) + obj.shape[1:])
        values[obj.dtype] = v
        return new_obj, values
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_obj[k], values = _recursive_write(v, values, sizes)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        new_obj = []
        for v in obj:
            new_v, values = _recursive_write(v, values, sizes)
            new_obj.append(new_v)
    else:
        raise ValueError("Unknown type `%s`" % type(obj))
    return new_obj, values

