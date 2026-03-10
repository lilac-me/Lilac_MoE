import logging
from typing import Optional, List, Callable
from math import log2
from datetime import timedelta
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
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# Expert data parallel group that the current rank belongs to.
_EXPERT_DATA_PARALLEL_GROUP = None
_EXPERT_DATA_PARALLEL_GROUP_GLOO = None

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


def get_nccl_options(pg_name: str, nccl_comm_cfgs: dict):
    """
    Set the NCCL process group options.

    Args:
        pg_name: process group name.
        ncc_comm_cfgs: nccl communicator configurations.
    """
    
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options(
            is_high_priority_stream=nccl_comm_cfgs[pg_name].get("is_high_priority_stream", False)
        )
        return nccl_options
    else:
        return None


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
    
    def decompose(index, shape, stride=None):
        """
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will be used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1] is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        )
        return idx
    
    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmasked for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            # combine the indices from masked and unmasked to get the global rank.
            global_rank = inner_product(decomposed_rank_idx, masked_stride) + inner_product(decomposed_group_idx, unmasked_stride)
            rank.append(global_rank)
        ranks.append(rank)
    return ranks


class RankGenerator(object):
    """
    A class for generating rank groups for different modes of parallelism.
    """
    def __init__(
        self, tp: int, cp: int, ep: int, dp: int, pp: int, order: str, rank_offset: int = 0
    )-> None:
        # tp-cp-dp-pp for Dense/attn while etp-ep-edp-pp for MoE/expert layers.
        assert (
            cp == 1 or ep == 1
        ), "Both EP and CP > 1 is not allowed in one rank generator."
        
        self.tp = tp
        self.cp = cp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.rank_offset = rank_offset
        self.world_size = tp * cp * ep * dp * pp

        self.name_to_size = {
            'tp': tp,
            'cp': cp,
            'ep': ep,
            'dp': dp,
            'pp': pp,
        }
        self.order = order
        order = order.lower()

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but we have not"
                    "specified the order ({self.order})."
                )
            elif name not in order:
                order = order + '-' + name
        
        self.order = order
        self.order_size = []
        
        for token in order.split('-'):
            self.order_size.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        """
        Create a mask for the specified tokens based on the given order.

        Args:
            order: The order of parallelism (e.g., "tp-dp-pp").
            token: The specific parallelism types to include in the mask,
            separated by hyphens(e.g., "tp-pp").
        """
        ordered_token = order.split('-')
        token_list = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token_list:
            mask[ordered_token.index(t)] = True
        return mask
    
    def get_ranks(self, token: str):
        """
        Get rank group by input token.

        Args:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple types, we can use a hyphen '-' to
                separate them. For example, if we want to obtain the TP_DP group,
                the token should be 'tp-dp'.
        """
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, self.order_size, mask)
        if self.rank_offset > 0:
            for rank_group in ranks:
                for i in range(len(rank_group)):
                    rank_group[i] += self.rank_offset
        return ranks
    

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    order: str="tp-cp-ep-dp-pp",
    ranks_offset: int = 0,
    create_gloo_process_groups: bool = True,
    distributed_timeout_minutes: int = 30,
):
    """
    Initialize parallel groups for different modes of parallelism.

    Args:
        tensor_model_parallel_size (int, default=1):
            The number of devices to split individual tensors across.

        pipeline_model_parallel_size (int, default=1):
            The number of tensor parallel device groups to split the 
            Transformer layers across.

        context_parallel_size (int, default=1):
            The number of tensor parallel device groups to split the network input 
            sequence length across. Compute attention module requires tokens of full
            sequence length, so devices in a context paralle group need to communicate 
            with each other to exchange information of other seuqence chunks.

            Context parallelism partitions sequence length, so it has no impact on weights,
            which means weights are duplicated among devices in a context paralle group. Hence,
            weight gradients all-reduce is required in backward.

        expert_model_parallel_size (int, default=1):
            The number of Mixture of Experts parallel devices in each expert parallel group.

        expert_tensor_parallel_size (int, default=tp_size):
            The number of devices to split individual tensors of expert.

        order (str, default="tp-dp-pp"):
            The rank initialization order fo parallelism.

        ranks_offset (int, default=0):
            The global rank offset for creating process groups, default is 0.

        create_gloo_process_groups (bool, default=True):
            Create gloo process groups if set to true. If set to false, gloo process groups
            are not created.

    Returns:
        A dictionary containing all the created process groups.
    """

    nccl_comm_cfgs = {}
    timeout = timedelta(distributed_timeout_minutes)

    assert torch.distributed.is_initialized(), "torch.distributed is not initialized."

    world_size = torch.distributed.get_world_size()
    model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size

    if world_size % model_size != 0:
        raise ValueError(
            f"World size {world_size} is not divisible by model size {model_size}."
        )
    
    data_parallel_size: int = world_size // model_size
    
    rank = torch.distributed.get_rank()

    # build dense/attn rank generator.
    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        cp=context_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        order=order,
        rank_offset=ranks_offset,
    )

    # build expert rank generator.
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tensor_model_parallel_size = (
        expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
    )

    if world_size % expert_tensor_model_parallel_size != 0:
        raise ValueError(
            f"World size {world_size} is not divisible by expert tensor model parallel size {expert_tensor_model_parallel_size}."
        )
    
    expert_data_parallel_size = world_size // expert_tensor_model_parallel_size

    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        cp=1,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        order=order,
        rank_offset=ranks_offset,
    )

    assert (
        order.endswith("pp")
        or pipeline_model_parallel_size == 1
        or expert_data_parallel_size == data_parallel_size
    ), "When not using pp-last rank ordering, the data parallel size of the attention and moe layers must be the same"

    assert decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks("pp"), \
        "The pp groups are expected to be the same for Non-MoE and MoE parts."
    

    # build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS

    for ranks in decoder_rank_generator.get_ranks("dp"):
        group = create_group(
            ranks=ranks,
            timeout=timeout,
            pg_options=get_nccl_options("dp", nccl_comm_cfgs),
            group_desc=f"DAPA_PARALLEL_GROUP",
        )

        if create_gloo_process_groups:
            group_gloo = create_group(
                ranks=ranks,
                timeout=timeout,
                backend="gloo",
                group_desc=f"DAPA_PARALLEL_GROUP_GLOO",
            )
        else:
            group_gloo = None
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GROUP_GLOO = group_gloo
            _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # build the context-parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    for ranks in decoder_rank_generator.get_ranks("cp"):
        group = create_group(
            ranks=ranks,
            timeout=timeout,
            pg_options=get_nccl_options("cp", nccl_comm_cfgs),
            group_desc=f"CONTEXT_PARALLEL_GROUP",
        )
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    # build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    global _MODEL_PARALLEL_GLOBAL_RANKS
    assert _MODEL_PARALLEL_GROUP is None, "Model parallel group is already initialized."
    for ranks in decoder_rank_generator.get_ranks("tp-pp"):
        group = create_group(
            ranks=ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp-pp", nccl_comm_cfgs),
            group_desc=f"MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group
            _MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # build the tensor-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, "Tensor model parallel group is already initialized."
    for ranks in decoder_rank_generator.get_ranks("tp"):
        group = create_group(
            ranks=ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp", nccl_comm_cfgs),
            group_desc=f"TENSOR_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # build the pipeline-parallel groups.
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, "Pipeline model parallel group is already initialized."
    for ranks in decoder_rank_generator.get_ranks("pp"):
        group = create_group(
            ranks=ranks,
            timeout=timeout,
            pg_options=get_nccl_options("pp", nccl_comm_cfgs),
            group_desc=f"PIPELINE_PARALLEL_GROUP",
        )
        if rank in ranks:
            _PIPELINE_PARALLEL_GROUP = group
            _PIPELINE_PARALLEL_GLOBAL_RANKS = ranks

    # build expert model parallel groups.
    global _EXPERT_MODEL_PARALLEL_GROUP
    global _EXPERT_MODEL_PARALLEL_GLOBAL_RANKS
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, "Expert model parallel group is already initialized."
    for ranks in expert_decoder_rank_generator.get_ranks("ep"):
        group = create_group(
            ranks=ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep", nccl_comm_cfgs),
            group_desc=f"EXPERT_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_MODEL_PARALLEL_GROUP = group
            _EXPERT_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # build expert tensor parallel groups.
    global _EXPERT_TENSOR_PARALLEL_GROUP
    assert _EXPERT_TENSOR_PARALLEL_GROUP is None, "Expert tensor model group is already initialized."
    for ranks in expert_decoder_rank_generator.get_ranks("tp"):
        group = create_group(
            ranks=ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep_tp", nccl_comm_cfgs),
            group_desc="EXPERT_TENSOR_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_TENSOR_PARALLEL_GROUP = ranks

    # build expert data parallel groups.
    global _EXPERT_DATA_PARALLEL_GROUP
    assert _EXPERT_DATA_PARALLEL_GROUP is None, "Expert data group is already initialized."
    global _EXPERT_DATA_PARALLEL_GROUP_GLOO
    assert _EXPERT_DATA_PARALLEL_GROUP_GLOO is None, "Expert data group-gloo is already initialized."
    for ranks in expert_decoder_rank_generator.get_ranks("dp"):
        group = create_group(
            ranks=ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep_dp", nccl_comm_cfgs),
            group_desc="EXPERT_DATA_PARALLEL_GROUP",
        )
        if create_gloo_process_groups:
            group_gloo = create_group(
                ranks, backend="gloo", group_desc="EXPERT_DATA_PARALLEL_GROUP_GLOO"
            )
        else:
            group_gloo = None
        if rank in ranks:
            _EXPERT_DATA_PARALLEL_GROUP = group
            _EXPERT_DATA_PARALLEL_GROUP_GLOO = group_gloo