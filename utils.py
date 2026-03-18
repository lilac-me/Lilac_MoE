import warnings

import torch

import parallel_state


def get_tensor_model_parallel_group_if_none(tp_group, is_expert=False, check_initialized=True):
    if not torch.distributed.is_initialized():
        return None
    if not parallel_state.is_initialized():
        return tp_group
    
    if tp_group is None:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            warnings.warn(
                "Warning: t_group is None, using default tp group. "
            )
        if is_expert:
            tp_group = parallel_state.get_expert_tensor_parallel_group(
                check_initialized=check_initialized
            )
        else:
            tp_group = parallel_state.get_tensor_model_parallel_group(
                check_initialized=check_initialized
            )
            
    return tp_group


def get_pg_size(group=None):
    """
    Get world size for distributed group.
    """
    if not torch.distributed.is_initialized() or group is None:
        return 1
    return group.size()


def get_pg_rank(group=None):
    """
    Get rank for distributed group.
    """
    if not torch.distributed.is_initialized() or group is None:
        return 0
    return group.rank()