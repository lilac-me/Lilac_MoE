from typing import Optional
import math

import torch
from torch import Tensor


def get_capacity(
    num_tokens: int, num_experts: int, capacity_factor: float, min_capacity: Optional[int] = None
) -> int:
    """
    Calculate the capacity of each expert.

    Args:
        num_tokens: num of the input tokens.
        num_experts: num of the experts.
        capacity_factor: capacity factor.
        min_capacity: minimum capacity, default to None.
    
    Returns:
        capacity of each expert.
    """
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity


def maybe_move_tensor_to_cpu(
    tensor: Tensor, as_numpy: bool = False, record_stream: bool = False
) -> Tensor:
    """
    Move the tensor to host if it is on the device
    
    Args:
        tensor: the tensor to move to host.
        as_numpy: whether to convert the tensors to a numpy array.
        record_stream: whether to record the stream of the tensor, to prevent
                       memory leak when the d2h data transfer is on a side stream.
    
    Returns:
        the tensor moved to host.
    """
    if torch.is_tensor(tensor) and tensor.is_cuda:
        cpu_tensor = tensor.to(torch.device("cpu"),non_blocking=True)
        if as_numpy:
            cpu_tensor = cpu_tensor.numpy()
        if record_stream:
            tensor.record_stream(torch.cuda.current_stream())
        tensor = cpu_tensor
    return tensor
