import logging
from typing import Optional, List, Callable
from math import log2
import einops

import torch

from moe_utils import GlobalMemoryBuffer


# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group(both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Context parallel group that the current rank belongs to.
_CONTEXT_PARALLEL_GROUP = None
# Expert parallel group that the current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
# Expert tensor parallel group that the current rank belongs to.
_EXPERT_TENSOR_PARALLEL_GROUP = None
# Tensor and expert parallel group that the current rank belongs to.
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Expert data parallel group that the current rank belongs to.
_EXPERT_DATA_PARALLEL_GROUP = None

# _EXPERT_MODEL denotes expert parallelism which splits the number of experts across the group
# _EXPERT_TENSOR denotes tensor parallelism of expert which parameters across the group
# _EXPERT_DATA denotes data parallelism of expert which replicates weight across the group

# A list of global ranks for each tensor model parallel group to ease calculation of the
# first local rank in the tensor model parallel group.
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each pipeline model parallel group to ease calculation of the
# source rank when broadcasting from the first or last pipeline stage.
_PIPELINE_MODEL_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each model parallel group to ease calculation of the
# first local rank in the model parallel group.
_MODEL_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel ranks.
_CONTEXT_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each expert parallel group to ease calculation of the
# first local rank in the expert parallel group.
_EXPERT_MODEL_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each expert tensor parallel group to ease calculation of the
# source rank when broadcasting weights from src to all other data parallel ranks.
_DATA_PARALLEL_GLOBAL_RANKS = None

# List of all process groups
# Used for updating the timeout for all process groups
# None represents the default process group
_global_process_group_list = None


def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    group_desc=None,
):
    """
    Create a ProcessGroup.
    """
    kwargs = {
        'ranks': ranks,
        'timeout': timeout,
        'backend': backend,
        'pg_options': pg_options,
        'group_desc': group_desc,
    }
    group = torch.distributed.new_group(**kwargs)
    global _global_process_group_list
    if _global_process_group_list is None:
        _global_process_group_list = [None]
    if torch.distributed.get_rank() in ranks:
        _global_process_group_list.append(group)
    return group


def generate_masked_orthogonal_rank_groups(
    world_size: int,
    parallel_size: List[int],
    mask: List[bool],
):
    """
    Communication sequence of Intra-Node is prioritized over Inter-Node.
    So note that the sequence of communication priority is `tp-cp-dp-pp` for Dense/attn
    while `etp-ep-edp-pp` for MoE/expert layers.

    Args:
        world_size: total number of devices in the distributed setup.
        parallel_size: a list of parallel sizes for each parallel group.
        mask: a list of ranks to be masked.

    Algorithm:
        two situations: [tp-cp-dp-pp] for Dense/attn
                        [etp-ep-edp-pp] for MoE/expert layers
        `global_rank` is we want to put in the process group.
        local_rank satisfies the following equations:
            global_rank = tp_rank + cp_rank * tp_size + dp_rank * tp_size * cp_size + pp_rank * tp_size * cp_size * dp_size
                tp_rank in [0, tp_size)
                cp_rank in [0, cp_size)
                dp_rank in [0, dp_size)
                pp_rank in [0, pp_size)
            
            global_rank = etp_rank + ep_rank * etp_size + edp_rank * etp_size * ep_size + pp_rank * etp_size * ep_size * edp_size
                etp_rank in [0, etp_size)
                ep_rank in [0, ep_size)
                edp_rank in [0, edp_size)
                pp_rank in [0, pp_size)
        if some process group is masked or None then xx_size=1 and the corresponding xx_rank is always 0.

        Let's take tp-dp-pp as an example while world_size=8, parallel_size=[2, 2, 2], mask=[False, False, False].
        That, the number of tp_group is 8/2 = 4, so do the dp_group and pp_group, here cp_size=1, cp_rank=0 so we 
        have: 
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size
                tp_rank in [0, 2)
                dp_rank in [0, 2)
                pp_rank in [0, 2)
        We want to get the each process group of tp:
            fix dp_rank=0, pp_rank=0, loop tp_rank in [0, 2) to get the 1st tp_group:
            global_rank = 0 + 0 * 2 + 0 * 2 * 2 = 0
            global_rank = 1 + 0 * 2 + 0 * 2 * 2 = 1

            fix dp_rank=1, pp_rank=0, loop tp_rank in [0, 2) to get the 2nd tp_group:
            global_rank = 0 + 1 * 2 + 0 * 2 * 2 = 2
            global_rank = 1 + 1 * 2 + 0 * 2 * 2 = 3

            fix dp_rank=0, pp_rank=1, loop tp_rank in [0, 2) to get the 3rd tp_group:
            global_rank = 0 + 0 * 2 + 1 * 2 * 2 = 4
            global_rank = 1 + 0 * 2 + 1 * 2 * 2 = 5

            fix dp_rank=1, pp_rank=1, loop tp_rank in [0, 2) to get the 4th tp_group:
            global_rank = 0 + 1 * 2 + 1 * 2 * 2 = 6
            global_rank = 1 + 1 * 2 + 1 * 2 * 2 = 7
        Then we get the tp_groups: [[0, 1], [2, 3], [4, 5], [6, 7]].

        Calculate dp_groups in a similar way.
            fix tp_rank=0, pp_rank=0, loop dp_rank in [0, 2) to get the 1st dp_group:
            global_rank = 0 + 0 * 2 + 0 * 2 * 2 = 0
            global_rank = 0 + 1 * 2 + 0 * 2 * 2 = 2

            fix tp_rank=1, pp_rank=0, loop dp_rank in [0, 2) to get the 2nd dp_group:
            global_rank = 1 + 0 * 2 + 0 * 2 * 2 = 1
            global_rank = 1 + 1 * 2 + 0 * 2 * 2 = 3

            fix tp_rank=0, pp_rank=1, loop dp_rank in [0, 2) to get the 3rd dp_group:
            global_rank = 0 + 0 * 2 + 1 * 2 * 2 = 4
            global_rank = 0 + 1 * 2 + 1 * 2 * 2 = 6

            fix tp_rank=1, pp_rank=1, loop dp_rank in [0, 2) to get the 4th dp_group:
            global_rank = 1 + 0 * 2 + 1 * 2 * 2 = 5
            global_rank = 1 + 1 * 2 + 1 * 2 * 2 = 7
        Then we get the dp_groups: [[0, 2], [1, 3], [4, 6], [5, 7]].

        We can get the pp_groups in a similar way.
            fix tp_rank=0, dp_rank=0, loop pp_rank in [0, 2) to get the 1st pp_group:
            global_rank = 0 + 0 * 2 + 0 * 2 * 2 = 0
            global_rank = 0 + 0 * 2 + 1 * 2 * 2 = 4

            fix tp_rank=1, dp_rank=0, loop pp_rank in [0, 2) to get the 2nd pp_group:
            global_rank = 1 + 0 * 2 + 0 * 2 * 2 = 1
            global_rank = 1 + 0 * 2 + 1 * 2 * 2 = 5

            fix tp_rank=0, dp_rank=1, loop pp_rank in [0, 2) to get the 3rd pp_group:
            global_rank = 0 + 1 * 2 + 0 * 2 * 2 = 2
            global_rank = 0 + 1 * 2 + 1 * 2 * 2 = 6

            fix tp_rank=1, dp_rank=1, loop pp_rank in [0, 2) to get the 4th pp_group:
            global_rank = 1 + 1 * 2 + 0 * 2 * 2 = 3
            global_rank = 1 + 1 * 2 + 1 * 2 * 2 = 7
        Then we get the pp_groups: [[0, 4], [1, 5], [2, 6], [3, 7]].
    """
    
    def prefix_product(a: List[int], init=1) -> List[int]:
        res = [init]
        for v in a:
            init = init * v
            res.append(init)
        return res
    
    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])
    