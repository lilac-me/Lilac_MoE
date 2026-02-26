from typing import Optional, Tuple, Callable
import math
from functools import reduce
import operator
from contextlib import nullcontext

import torch
from torch import Tensor


_GLOBAL_MEMORY_BUFFER = None
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"


class GlobalMemoryBuffer:
    """
    Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name are not used concurrently.
    """
    def __init__(self):
        self.buffer = {}
    
    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context: Optional[Callable] = None):
        """
        Return a sub-tensor from the self.buffer for the given shape.
        """
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            mem_alloc_context = mem_alloc_context if mem_alloc_context else nullcontext
            with mem_alloc_context():
                self.buffer[(name, dtype)] = torch.empty(
                    required_len,
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                )
        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def get_global_memory_buffer():
    """
    Get the global memory buffer.
    """
    global _GLOBAL_MEMORY_BUFFER
    if _GLOBAL_MEMORY_BUFFER is None:
        _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()
    return _GLOBAL_MEMORY_BUFFER


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


def permute(
    tokens: Tensor,
    routing_map: Tensor,
    probs: Optional[Tensor] = None,
    num_out_tokens: Optional[Tensor] = None,
    drop_and_pad: bool = False,
    tokens_per_expert: Optional[Tensor] = None,
    align_size: int = -1,
) -> Tuple[
    Tensor,
    Optional[Tensor],
    Tensor,
    Optional[Tensor],
    Optional[Tensor],
]:
    """
    Permute the tokens and probs bases on the mask.

    Tokens with the same designated expert will be grouped together.
    The shape of the mask is [tokens, num_experts], it indicates which expert were selected
    by each token.

    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.

    If the fused permute and pad kernel is available, it will pad the tokens to the align_size
    and return the padded permute tokens, pad_offsets and padded tokens per expert.

    Args:
        tokens: the input token tensor, [num_tokens, hidden_size]
        routing_map: the sparse token to expert mapping, [num_tokens, num_experts]
        probs: the prob tensor, [num_tokens, num_experts]
        num_out_tokens: the number of output tokens. If None, it's set to the number of input tokens.
        drop_and_pad: whether or not the token dispatcher uses token-drop and pads the number
                      of tokens to the expert capacity. If set to true, routing_map has a fixed
                      number of non-zeros in each column.
        tokens_per_expert: tensor of shape [num_experts] containing actual token counts per expert.
        align_size: the alignment size for the input tensor for fp8 or fp4.
    
    Returns:
        Tuple[
            Tensor,
            Optional[Tensor],
            Tensor,
            Optional[Tensor],
            Optional[tensor],
        ]:
            the permuted tokens
            (optional) permuted_probs
            sorted_indices,
            (optional) pad_offsets
            (optional) padded_tokens_per_expert
    """
    num_tokens, hidden_size = tokens.shape # [S*B*EP, H] -> [num_tokens, hidden_size]
    num_experts = routing_map.size()[1] # [num_local_experts]
    permuted_probs = None
    if drop_and_pad and num_out_tokens is not None:
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_local_experts] -> [num_local_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first capacity number of indices.
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
            :, :capacity
        ].contiguous()
        # flatten from [num_local_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1) # [num_local_experts*capacity]

        if probs is not None:
            # [num_tokens, num_local_experts] -> num_local_experts * num_tokens
            probs_T_1D = probs.T.contiguous().view(-1)
            # get 1D indices of the probs selected by routing_map
            # indices_dim0 is expert range for shape of [num_local_experts]
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            # expert_id * num_tokens + token_id
            indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
    else:
        # mask [num_tokens, num_local_experts] -> [num_local_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()
        # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        ) # [num_local_experts, num_tokens]
        sorted_indices = token_indices.masked_select(routing_map)
        
        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)
    
    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices, None, tokens_per_expert


def unpermute(
    permuted_tokens: Tensor,
    sorted_indices: Tensor,
    restore_shape: torch.Size,
    probs: Optional[Tensor] = None,
    routing_map: Optional[Tensor] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
    pad_offsets: Optional[Tensor] = None,
) -> Tensor:
    """
    Restore the original order of tokens after permutation. If probs is not None, it will
    also apply them to the tokens before restoring the order.

    When drop_and_pad=Ttue, the tensors will have the following properties:
        - in routing_map, the number of non-zeros in each column equals to expert capacity.
        - the size of sorted_indices equals to num_experts * capacity, each splits of 'capacity'
          contains the indices of tokens routed to an expert.
    This function exploits these features to use ops that support cuda graph.

    Args:
        permuted_tokens: the permuted token tensor.
        sorted_indices: the indices used to sort the tokens.
        restore_shape: the shape of the unpermuted tensor.
        probs: the permuted probs tensor.
        routing_map: token to expert mapping, shape [num_tokens, num_experts]
        fused: whether to use fused unpermute function.
        drop_and_pad: whether or not the token dispatcher uses token-drop and pads the number of
                      tokens to the expert capacity.
        pad_offsets:
            tensor of per-expert cumulative padding offsets used to remove padding added during
            permutation. This is the fourth output of 'moe_permute_and_pad_with_probs' and is 
            required when unpermuting padded outputs. Default to None

    Returns:
        the tokens restored to their original order.
    """
    _, hidden_size = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size()[1]
            num_permuted_tokens = sorted_indices.size()[0]
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size()[0]

            # [num_unpermuted_tokens, num_experts] -> num_experts * num_unpermuted_tokens
            probs_T_1D = probs.T.contiguous().view(-1)
            # get 1D indices of the probs selected by routing_map
            # indices_dim0 is expert range
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        # Here may promote permuted_tokens to higher precision (fp32/fp64) if probs is in 
        # higher precision due to moe_router_dtype being enabled. This can lead to additional
        # device memory usage. Use --moe-permute-fusion flag to avoid this extra memory allocation.
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)
    
    # Create an output tensor filled with zeros.
    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    if torch.are_deterministic_algorithms_enabled():
        # use index_add which is deterministic the deterministic algorithm are enabled and
        # is cuda graph compatible
        output_tokens = torch.zeros(
            restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
        )
        # index_add is deterministic when torch.use_deterministic_algorithms(True) is set 
        # and cuda graph compatible unlike scatter_add
        output_tokens.index_add_(0, sorted_indices, permuted_tokens)
    else:
        # scatter add the permuted_tokens back to the original positions
        output_tokens.scatter_add_(
            0, sorted_indices.unsqueeze(1).expand(-1, hidden_size), permuted_tokens
        )
    return output_tokens.to(dtype=input_dtype)


def sort_chunks_by_idxs(
    input: Tensor,
    split_sizes: Tensor,
    sorted_idxs: Tensor,
    probs: Optional[Tensor] = None,
    fused: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Split and sort the input tensor based on the split_sizes and sorted_indices.

    Args:
        input: the input tensor.
        split_sizes: the split sizes.
        sorted_idxs: the sorted indices.
        probs: the probs tensor. Default to None.
        fused: whether to use the fused version of the sorted_chunks_by_idxs function.
               Default to False.
    
    Returns:
        the sorted output tensor and permuted probs.
    """
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs.tolist()], dim=0)
    if probs is not None:
        probs = torch.split(probs, split_sizes.tolist(), dim=0)
        permuted_probs = torch.cat([probs[i] for i in sorted_idxs.tolist()], dim=0)
    else:
        permuted_probs = None
    return output, permuted_probs


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
