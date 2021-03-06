"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import io
import time

import torch
import torch.distributed as dist

from IPython import embed


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = io.BytesIO()
    torch.save(data, buffer)
    buffer.seek(0)
    storage = torch.ByteStorage.from_buffer(buffer.read())
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = io.BytesIO(tensor.cpu().numpy().tobytes()[:size])
        data_list.append(torch.load(buffer))

    return data_list


def broadcast(data, src=None):
    assert src is not None

    world_size = get_world_size()
    if world_size == 1:
        return data

    # serialized to a Tensor
    buffer = io.BytesIO()
    torch.save(data, buffer)
    buffer.seek(0)
    storage = torch.ByteStorage.from_buffer(buffer.read())
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    dist.broadcast(local_size, src=src)

    # build actual tensor to be broadcast to
    if get_rank() != src:
        tensor = torch.ByteTensor(size=(local_size,)).to("cuda")
    dist.broadcast(tensor, src=src)
    buffer = io.BytesIO(tensor.cpu().numpy().tobytes()[:local_size])
    data = torch.load(buffer)

    return data
