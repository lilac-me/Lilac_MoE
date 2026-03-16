import torch


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