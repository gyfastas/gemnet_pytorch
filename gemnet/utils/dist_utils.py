import os
import torch
from torch import distributed as dist
import logging

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0

def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1

def barrier():
    if get_world_size() > 1:
        dist.barrier()

def check_and_initialize(world_size, rank, **kwargs):
    if world_size > 1 and not dist.is_initialized():
        if rank == 0:
            logging.info("Initializing distributed")
        dist.init_process_group("nccl", init_method="env://", **kwargs)

def get_device(world_size, rank):
    gpu = rank % world_size
    return torch.device(gpu)