from typing import Optional, Tuple
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


def permute(
    tokens: Tensor,
    routing_map: Tensor,
    probs: Optional[Tensor] = None,
    num_out_tokens: Optional[Tensor] = None,
    fused: bool = False,
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
        fused: whether use the fused permute function.
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
    num_tokens, hidden_size = tokens.shape
    num_experts = routing_map.size()[1]
    permuted_probs = None
    if drop_and_pad and num_out_tokens is not None:
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first capacity number of indices.
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
            :, :capacity
        ].contiguous()
        # flatten from [num_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1)

        if probs is not None:
            # [num_tokens, num_experts] -> num_experts * num_tokens
            probs_T_1D = probs.T.contiguous().view(-1)
            # get 1D indices of the probs selected by routing_map
            # indices_dim0 is expert range
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            # expert_id * num_tokens + token_id
            indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
    else:
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()
        # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map)
        
        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)
    
    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices, None, tokens_per_expert


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
