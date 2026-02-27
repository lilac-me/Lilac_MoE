from typing import Optional, List
from abc import abstractmethod

import torch
from torch import Tensor

from moe_utils import (
    get_capacity,
    maybe_move_tensor_to_cpu,
    permute,
    unpermute,
    sort_chunks_by_idxs,
)
from mapping import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)


"""
    We use the following notation throughout the codebase:
    H: hidden size
    S: sequence length
    B: batch size
    TP: tensor model parallel size
    EP: expert model parallel size
    num_local_tokens: S/TP*B
    num_global_tokens: num_local_tokens*TP*EP = S/TP*B*TP*EP = S*B*EP
"""

class MoETokenDispatcher:
    """
    A module that dispatches tokens to experts for MoE layers. It supports both local and global dispatching.
    Local dispatching means that each GPU only processes the tokens assigned to it, while global dispatching means that all GPUs process all tokens.
    The dispatcher also supports different token formats, such as packed sequences and regular sequences.
    """

    def __init__(self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None) -> None:
        """
        Initialize the MoE Token Dispatcher.

        Args:
            config (TransformerConfig): Configuration for the MoE layer.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        self.config = config
        self.shared_experts = Optional[SharedExpertMLP] = None

        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor model parallel group in this module.
        self.tp_group = pg_collection.expt_tp
        self.tp_ep_group = pg_collection.tp_ep

        self.tp_size = utils.get_pg_size(self.tp_group)
        self.tp_rank = utils.get_pg_rank(self.tp_group)
        self.ep_size = utils.get_pg_size(self.ep_group)

    @abstractmethod
    def dispatch_preprocess(
        self, tokens: Tensor, routing_map: Tensor, probs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Prepare tokens for dispatch without inter-device communication.

        This method should handle all local computations like tensor rearrangements and
        metadata extraction before the main communication step.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
        tokens (torch.Tensor): Input tokens.
        routing_map (torch.Tensor): Token to expert mapping tensor.
        probs (torch.Tensor): The routing probability tensor, [num_tokens, num_experts].

        Returns:
            A tuple of preprocessed tokens and probabilities.
        """
        raise NotImplementedError("dispatch_preprocess function not implemented")
    
    @abstractmethod
    def token_dispatch(self, hidden_states: Tensor, probs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Dispatch tokens to expert devices using communication.

        This method performs the main communication (e.g., All-to-All) to send
        tokens to the devices where their assigned expert resides.

        Args:
            hidden_states (Tensor): Preprocessed token hidden states ready for dispatch.
            probs (Tensor): Preprocessed probabilities for each token-expert pair.

        Returns:
            A tuple of dispatched tokens and the corresponding probabilities after dispatch.
        """
        raise NotImplementedError("token_dispatch function not implemented")
    
    @abstractmethod
    def dispatch_postprocess(self, hidden_states: Tensor, probs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Performs local processing after token dispatch communication.

        This method handles post-communication tasks like token reordering and preparing metadata
        for the expert forward pass.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            hidden_states (torch.Tensor): Dispatched hidden states.
            probs (torch.Tensor): Dispatched probabilities.

        Returns:
            A tuple containing the permuted tokens for experts, the number of
            tokens per expert, and the permuted probabilities.
        """
        raise NotImplementedError("dispatch_postprocess function not implemented")
    
    @abstractmethod
    def combine_preprocess(self, hidden_states: Tensor) -> Tensor:
        """
        Preprocess experts outputs for the combine step.

        This method performs local computations on expert outputs before the communication
        step for combining the outputs back to the original token order.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.
        
        Args:
            hidden_states (torch.Tensor): The output hidden states from the experts.

        Returns:
            A tensor of preprocessed hidden states ready for the combine communication step.
        """
        raise NotImplementedError("combine_preprocess function not implemented")
    
    @abstractmethod
    def token_combine(self, hidden_states: Tensor) -> Tensor:
        """
        Combine expert outputs across devices using communication.

        This method performs the main communication (e.g., All-to-All, Reduce-Scatter) to gather
        the processed tokens from the experts and combine them back to the original order.

        Args:
            hidden_states (Tensor): Preprocessed hidden states from experts ready for combination.
        
        Returns:
            A tensor of combined hidden states in the original token order.
        """
        raise NotImplementedError("token_combine function not implemented")
    
    @abstractmethod
    def combine_postprocess(self, hidden_states: Tensor) -> Tensor:
        """
        Postprocess combined hidden states after the combine communication step.

        This method performs post-communication tasks like unpermuting and reshaping 
        to restore the original tensor structure.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            hidden_states (torch.Tensor): Combined hidden states after communication.
        
        Returns:
            The final output tensor.
        """
        raise NotImplementedError("combine_postprocess function not implemented")
    
    def set_shared_expert(self, shared_experts: SharedExpertMLP) -> None:
        """
        Set the shared expert MLP for this token dispatcher.
        """
        assert self.config.moe_shared_expert_overlap
        self.shared_experts = shared_experts
    

class MoEAllgatherTokenDispatcher(MoETokenDispatcher):
    """
    AllGather based token Dispatcher.
    Note that this allgather spans the communication domain to TP*EP.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,   
    ) -> None:
        """
        Initialize the MoEAllgatherTokenDispatcher based on the token dispatcher.

        Args:
            num_lcoal_expert (int): The number of local experts.
            local_expert_indices (List[int]): The list of local expert indices.
            config (TransformerConfig): Configuration for the MoE layer.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config, pg_collection)
        self.num_local_experts = num_local_experts
        assert self.num_local_experts > 0, "num_local_experts should be greater than 0"
        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) > 0, "local_expert_indices should not be empty"
        self.router_topk = config.moe_router_topk
        self.bias = config.add_bias_linear

        # self.global_local_map: 2D tensor. A mask of mapping between global and local expert tokens
        # where each element is True if it's between the local_expert_indices. Only useful
        # when cross device token permutation is enabled and *AllGather* is performed.

    def dispatch_preprocess(
        self, hidden_states: Tensor, routing_map: Tensor, probs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Reshape hidden states and caches the routing map.

        Args:
            hidden_states (Tensor): 3D [S/TP, B, H].
            routing_map (Tensor): 2D [S/TP*B, num_experts]
            probs (Tensor): 2D [S/TP*B, num_experts]
        """
        self.hidden_shape = hidden_states.shape # [S/TP, B, H]
        # [S/TP, B, H] -> [S/TP*B, H]
        hidden_states = hidden_states.reshape(-1, self.hidden_shape[-1])
        self.routing_map = routing_map
        return hidden_states, probs
    
    def token_dispatch(self, hidden_states: Tensor, probs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Gather tokens from all TP*EP ranks using AllGather.

        Args:
            hidden_states (Tensor): 2D [S/TP*B, H].
            probs (Tensor): 2D [S/TP*B, num_experts].
        """
        # Permute the tokens across the expert parallel devices.
        if self.tp_size > 1 or self.ep_size > 1:
            ## local_expert_indices calculation.
            with torch.no_grad():
                # [num_local_tokens, num_experts] -> [num_global_tokens, num_experts] where
                # num_local_tokens = S/TP*B, num_global_tokens = S*B*EP
                self.routing_map = gather_from_sequence_parallel_region(
                    self.routing_map, self.tp_ep_group
                )
            ## local_probs calculation.
            # max_probs: [S/TP*B, num_experts] -> global_probs: [S*B*EP, num_experts]
            probs = gather_from_sequence_parallel_region(probs, self.tp_ep_group)
            # [S/TP*B, H] -> [(S/TP)*B*(TP*EP), H] -> [S*B*EP, H]
            hidden_states = gather_from_sequence_parallel_region(hidden_states, self.tp_ep_group, use_global_buffer=True)
        
        return hidden_states, probs
    
    def dispatch_postprocess(self, hidden_states: Tensor, probs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        After gatherting these tokens, this method identifies the tokens for local experts and
        permutes the tokens to group them by expert for efficient expert processing.

        Args:
            hidden_states (Tensor): 2D [S*B*EP, H].
            probs (Tensor): 2D [S*B*EP, num_experts].
        """
        self.hidden_shape_before_permute = hidden_states.shape # [S*B*EP, H]

        # Note that the self.routing_map shape is [S*B*EP, num_experts], and the local experts 
        # are a subset of the global experts, so we can slice the routing map to get the local map.
        # Whose shape is [S*B*EP, num_local_experts]
        self.local_map = self.routing_map[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # Probs of global token assignment to local experts. [S*B*EP, num_local_experts]
        self.local_probs = probs[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()

        tokens_per_expert = self.local_map.sum(dim=0).long().cpu() # [num_local_experts]

        permuted_local_hidden_states, _, self.reversed_local_input_permutation_mapping, _, _ = permute(
            hidden_states, # [S*B*EP, H]
            self.local_map, # [S*B*EP, num_local_experts] -> [num_local_tokens, num_local_experts]
            num_out_tokens=tokens_per_expert.sum().item(),
        )
        # permuted_local_hidden_states: [num_local_tokens, H]
        # self.reversed_local_input_permutation_mapping: [num_local_tokens]

        self.local_probs = self.local_probs.T.contiguous().masked_select(
            self.local_map.T.contiguous()
        )
        # self.local_probs.T: [num_local_experts, S*B*EP]
        # self.local_map.T: [num_local_experts, S*B*EP]
        self.routing_map = None
        return permuted_local_hidden_states, tokens_per_expert, self.local_probs
    
    def combine_preprocess(self, hidden_states: Tensor) -> Tensor:
        """
        Reverses token permutation to restore original ordering before reduction operations.

        This method unpermutes the expert outputs using the cached permutation mapping
        from the dispatch phase. The unpermutation operation restores tokens to their
        original sequence positions, preparing them for the subsequent reduction scatter
        operation that will aggregate results across ranks.
        """
        unpermuted_local_hidden_states = unpermute(
            hidden_states, # [num_local_tokens, H]
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.local_map,
        )
        # unpermuted_local_hidden_states: [S*B*EP, H]
        return unpermuted_local_hidden_states
    
    def token_combine(self, hidden_states: Tensor) -> Tensor:
        """
        Reduce-Scatter the unpermuted tokens back to the original sequence parallel region.

        This method performs the ReduceScatter communication operation to collect expert
        outputs from their processing ranks and redistribute tokens back to the ranks that
        originally held them. This completes the expert processing
        communication pattern and prepares tokens for final unpermutation.

        Args:
            hidden_states (Tensor): 2D [S*B*EP, H].
        """
        if self.tp_size > 1 or self.ep_size > 1:
            # [S*B*EP, H] -> [S/TP*B, H]
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states.to(self.local_probs.dtype), self.tp_ep_group
            ).to(hidden_states.dtype)
        return hidden_states
    
    def combine_postprocess(self, hidden_states: Tensor) -> Tensor:
        """
        Restoring the original tensor shape..

        Args:
            hidden_states (Tensor): 2D [num_local_tokens, H].
        """
        return hidden_states.view(self.hidden_shape) # [S/TP*B, H] -> [S/TP, B, H]


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AllToAll based token Dispatcher.
    
    The workflow of AlltoAll token dispatcher is as following:
    (1) preprocess: calculate necessary metadata for communication and permute.
    (2) dispatch preprocess: permute tokens.
    (3) token dispatch: A2A(EP).
    (4) dispatch postprocess: AG(TP) -> sort_chunk(if num_local_experts > 1).
    (5) combine preproess: sort_chunk(if num_local_experts > 1) -> RS(TP).
    (6) token combine: A2A(EP).
    (7) combine postprocess: unpermute tokens
    """

    # d2h copies are performed on this stream for overlapping with the main stream.
    d2h_stream = None

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,   
    ) -> None:
        """
        Initialize the MoEAlltoAllTokenDispatcher based on the token dispatcher.

        Args:
            num_lcoal_expert (int): The number of local experts.
            local_expert_indices (List[int]): The list of local expert indices.
            config (TransformerConfig): Configuration for the MoE layer.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config, pg_collection)
        self.num_local_experts = num_local_experts
        assert config.num_moe_experts is not None
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert."
        self.local_expert_indices = local_expert_indices
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices."
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be contiguous."
        
        # [ep_size]. Represents the number of tokens sent by the current rank to other EP ranks.
        self.input_splits = None
        # [ep_size]. Represents the number of tokens received by the current rank from other EP ranks.
        self.output_splits = None
        # [tp_size]. Represents the number of tokens received by the current rank from other TP ranks.
        self.output_splits_tp = None
        self.permute_idx_device = torch.device("cuda") if config.moe_permute_fusion else "cpu"
        # [num_experts * tp_size] = [num_local_experts * ep_size * tp_size].
        input_chunk_idxs = torch.arange(
            self.num_experts * self.tp_size, device=self.permute_idx_device
        )
        # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        # Token drop and padding
        # Drop and pad the input to capacity.
        self.drop_and_pad = config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert config.moe_expert_capacity_factor is not None
            self.moe_expert_capacity_factor = config.moe_expert_capacity_factor
        self.capacity = None

        # A cuda stream synchronized is needed in during token permutation in some cases,
        # because there are several non-blocking d2h data transfers called at 'self.cuda_d2h_point'.
        # The synchronization happens at 'self.cuda_sync_point', which is decided based on the
        # MoE and parallel settings. Valid points are 'before_permutation_1', 'before_ep_alltoall',
        # 'before_permutation_2', 'before_finish' and 'no_sync'.
        self.cuda_sync_point = "no_sync"
        self.cuda_sync_point_priority = {
            "before_permutation_1": 0,
            "before_ep_alltoall": 1,
            "before_permutation_2": 2,
            "before_finish": 3,
            "no_sync": 4,
        }
        self.cuda_d2h_point = "before_permutation_1"
        if MoEAlltoAllTokenDispatcher.d2h_stream is None:
            MoEAlltoAllTokenDispatcher.d2h_stream = torch.cuda.Stream()

        self.shared_experts = None

    def preprocess(self, routing_map: Tensor) -> Tensor:
        """
        Preprocess the token routing map for All-to-All communication and token permutation.
        
        This method computes the number of tokens assigned to each expert based on the routing_map.
        It also initilizes necessary data structures for All-to-All communication, such as input
        and output splits, and the mapping between global tokens and local experts. This method should
        not call any d2h data coping due to performance consideration. The necessary d2h copies are
        made on the 'self.cuda_d2h_stream' at 'self.cuda_d2h_point'.

        Args:
            routing_map (Tensor): 2D [S/TP*B, num_experts]. The token to expert mapping tensor.
        
        Returns:
            A tensor with the number of tokens for each local expert.
        """
        if self.drop_and_pad:
            # Drop and pad the input to capacity.
            num_tokens = routing_map.size(0) * config.moe_router_topk
            self.capacity = get_capacity(
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.moe_expert_capacity_factor,
            )
            self.num_out_tokens = self.capacity * self.num_experts
            # [num_local_experts] number of tokens processed by each expert.
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,),
                self.capacity * self.tp_size * self.ep_size,
                dtype=torch.long,
            )
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens
            # sent to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = torch.full(
                (self.num_experts * self.tp_size,),
                self.capacity,
                dtype=torch.long,
                device=self.permute_idx_device,
            )
            return num_tokens_per_local_expert
        
        # [num_experts], number of tokens assigned to each expert from the current rank's input.
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()

        if (
            config.moe_expert_capacity_factor is not None
            or config.moe_router_padding_for_quantization
        ):
            # When using token dropping or router padding, output size is dynamic.
            # Need to sync output size device->host before allocating output buffer.
            self.num_out_tokens = num_local_tokens_per_expert.sum()
            self._maybe_update_cuda_sync_point("before_permutation_1")
        else:
            # For dropless training, output size is static (num_tokens * topk)
            # No explicit synchronization is needed.
            self.num_out_tokens = routing_map.size(0) * config.
        if self.ep_size > 1 or self.tp_size > 1:
            # Calculate 'input_splits', 'output_splits' for alltoall/allgather in variable size.
            # [ep_size]. Represents the number of tokens sent by the current rank to other EP ranks.
            # [num_experts] -> [ep_size, num_local_experts] -> [ep_size]
            self.input_splits = num_local_tokens_per_expert.reshape(
                self.ep_size, self.num_local_experts
            ).sum(axis=1)
            # Gather the global distribution of tokens across ranks.
            # num_global_tokens_per_expert represents the number of tokens sent to each expert by all ranks.
            # [tp_size, ep_size, num_experts]
            num_global_tokens_per_expert = (
                gather_from_sequence_parallel_region(
                    num_local_tokens_per_expert, group=self.tp_ep_group,
                )
                .reshape(self.ep_size, self.tp_size, self.num_experts)
                .transpose(0, 1)
            )
            # [tp_size, ep_size, num_experts] -> [tp_size, ep_size, num_local_experts]
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()
            # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
            num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
            
            # [tp_size, ep_size] -> [ep_size]
            # self.output_splits represents the number of tokens received by the current rank from other EP ranks.
            self.output_splits = num_global_tokens_per_rank[self.tp_rank]
            # [tp_size, ep_size] -> [tp_size]
            # self.output_splits_tp represents the number of tokens received by the current rank from other TP ranks.
            self.output_splits_tp = num_global_tokens_per_rank.sum(axis=1)
            # [tp_size, ep_size, num_local_experts] -> [num_local_experts]
            num_local_tokens_per_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))

            # A synchronization is needed before expert parallel AlltoAll communication
            # to get the 'inputs_splits' and 'output_splits'  CPU values.
            self._maybe_update_cuda_sync_point("before_ep_alltoall")
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert
            # A synchronization is needed before the returns to get the 'num_tokens_per_local_expert' CPU values.
            self._maybe_update_cuda_sync_point("before_finish")
        
        if self.num_local_experts > 1:
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            )
            if not config.moe_permute_fusion:
                # A synchronization is needed before permutation 2 to get the 'num_global_tokens_per_local_expert' CPU values.
                self._maybe_update_cuda_sync_point("before_permutation_2")
        
        assert (
            self.cuda_sync_point_priority[self.cuda_d2h_point]
            <= self.cuda_sync_point_priority[self.cuda_sync_point]
        ), "cuda_sync_point must be after cuda_d2h_point."
        return num_local_tokens_per_expert
    
    def dispatch_preprocess(
        self, hidden_states: Tensor, routing_map: Tensor, probs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Prepare hidden states and probilities for dispatch.

        This method reshapes the hidden states, computes mcommunication metadata,
        and permutes the tokens and probilities before All-to-All communication.

        Args:
            hidden_states (Tensor): 3D [S/TP, B, H].
            routing_map (Tensor): 2D [S/TP*B, num_experts].
            probs (Tensor): 2D [S/TP*B, num_experts].
        
        Returns:
            A tuple of permuted hidden states and permuted probabilities ready for dispatch communication.
        """
        # Preprocess: Get the metadata for communication and, permutaiton and computation operations.
        self.hidden_shape = hidden_states.shape # [S/TP, B, H]
        self.routing_map = routing_map
        self.probs = probs
        assert probs.dim() == 2, "probs should be 2D [S/TP*B, num_experts]"
        assert routing_map.dim() == 2, "routing_map should be 2D [S/TP*B, num_experts]"
        assert routing_map.dtype == torch.bool, "routing_map should be a boolean tensor"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1]) # [S/TP*B, H]

        if config.moe_router_padding_for quantization:
            pad_multiple = get_align_size_for_quantization(self.config)
            if is_experimental_enabled() and self.config.moe_permute_fusion:
                self.routing_map = fused_pad_routing_map(self.routing_map, pad_multiple)
            else:
                self.routing_map = pad_routing_map(self.routing_map, pad_multiple)
        self.tokens_per_expert = self.preprocess(self.routing_map)

        if self.shared_experts is not None:
            self.shared_experts.pre_foward_comm(hidden_states.view(self.hidden_shape))

        # Permutation 1: input to AlltoAll input
        self.tokens_per_expert = self._maybe_d2h_and synchronize(
            "before_permutation_1", self.tokens_per_expert
        )
        self.hidden_shape_before_permute = hidden_states.shape
        (
            permuted_local_input_tokens,
            permuted_probs,
            self.reversed_local_input_permutation_mapping,
            _,
            _,
        ) = permute(
            hidden_states,
            self.routing_map,
            probs=probs,
            num_out_tokens=self.num_out_tokens,
            drop_and_pad=self.drop_and_pad,
        )
        return permuted_local_input_tokens, permuted_probs

    def token_dispatch(self, permuted_local_input_tokens: Tensor, permuted_probs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Perform All-to-All communication to dispatch tokens to expert devices.

        This method performs all-to-all communication step to dispatch tokens across expert parallel
        ranks. It synchronizes the metadata at the appropriate point before performing the communication.

        Args:
            permuted_local_input_tokens (Tensor): 2D [num_local_tokens, H]. The permuted hidden states ready for dispatch.
            permuted_probs (Tensor): 2D [num_local_tokens, num_local_experts]. The permuted probabilities ready for dispatch.
        
        Returns:
            A tuple of hidden states and probabilities after dispatch communication.
        """
        # Perform expert parallel AlltoAll communication.
        self.tokens_per_expert = self._maybe_d2h_and_synchronize(
            "before_ep_alltoall", self.tokens_per_expert
        )
        global_input_tokens = all_to_all(
            permuted_local_input_tokens, self.input_splits, self.output_splits, group=self.ep_group
        )
        global_probs = all_to_all(
            permuted_probs, self.input_splits, self.output_splits, group=self.ep_group
        )
        return global_input_tokens, global_probs

    def dispatch_postprocess(self, global_input_tokens: Tensor, global_probs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Postprocess the dispatched tokens after All-to-All communication.

        This method involves an All-Gather in the tensor parallel dimension and sorting tokens
        by expert if there are multiple local experts.

        Args:
            global_input_tokens (Tensor): 2D [S*B*EP, H]. The hidden states after dispatch communication.
            global_probs (Tensor): 2D [S*B*EP, num_local_experts]. The probabilities after dispatch communication.
        
        Returns:
            A tuple of hidden states, number of tokens per expert, and probabilities after postprocessing.
        """
        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)

        if self.tp_size >1:
            if self.output_splits_tp is None:
                output_split_sizes = None
            else:
                output_split_sizes = self.output_splits_tp.tolist()
            global_input_tokens = gather_from_sequence_parallel_region(
                global_input_tokens, self.tp_group, output_split_sizes=output_split_sizes
            )
            global_probs = gather_from_sequence_parallel_region(
                global_probs, self.tp_group, output_split_sizes=output_split_sizes
            )

        # Permutation 2: Sort tokens by local experts.
        self.tokens_per_expert = self._maybe_d2h_and_synchronize(
            "before_permutation_2", self.tokens_per_expert
        )
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                global_input_tokens = (
                    global_input_tokens.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_input_tokens.shape[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
                global_probs = (
                    global_probs.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_probs.shape[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                global_input_tokens, global_probs = sort_chunks_by_idxs(
                    global_input_tokens,
                    self.num_global_tokens_per_local_expert.ravel(),
                    self.sort_input_by_local_experts,
                    probs=global_probs,
                )
        tokens_per_expert = self._maybe_d2h_and_synchronize(
            "before_finish", self.tokens_per_expert
        )
        self.tokens_per_expert = None
        return global_input_tokens, tokens_per_expert, global_probs
    
    def combine_preprocess(self, hidden_states: Tensor) -> Tensor:
        """
        Prepares hidden states for token combination after expert computations.

        This may involves un-sorting tokens and Reduce-Scatter in the tensor parallelism
        """
        # Unpermutation2: Unsort tokens by local experts.
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                hidden_states = (
                    hidden_states.view(
                        self.num_local_experts,
                        self.tp_size * self.ep_size,
                        self.capacity,
                        *hidden_states.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                hidden_states, _ = sort_chunks_by_idxs(
                    hidden_states,
                    self.num_global_tokens_per_local_expert.T.ravel(),
                    self.restore_output_by_local_experts,
                )
        
        if self.tp_size > 1:
            if self.output_splits_tp is None:
                input_split_sizes = None
            else:
                input_split_sizes = self.output_splits_tp.tolist()
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states.to(self.probs.dtype),
                group=self.tp_group,
                input_split_sizes=input_split_sizes
            ).to(hidden_states.dtype)
        
        return hidden_states
    
    def token_combine(self, hidden_states: Tensor, async_finish: bool = True, allocate_on_comm_stream: bool = True) -> Tensor:
        """
        Executes fused unpermutation and communication using DeepEP kernel.

        This method performs the inverse AlltoAll communication operation to collect expert
        outputs from their processing ranks and redistribute them back to the ranks that originally
        held the corresponding tokens. This completes the expert processing communication pattern
        and prepares tokens for final unpermutation.

        Args:
            hidden_states: expert outputs ready for conbination.
            async_finish: whether to use asynchronous communication completion.
            allocate_on_comm_stream: whether to allocate buffers on communication stream
        
        Returns: tokens after All-t-oAll communication for combining.
        """
        # Perform expert parallel AlltAll communication
        # hidden_states: [SEQL, H] -> [H, SEQL/TP]
        permutated_local_input_tokens = all_to_all(
            self.ep_group, hidden_states, self.input_splits, self.output_splits
        )
        return permutated_local_input_tokens
    
    def combine_postprocess(self, permutated_local_input_tokens: Tensor) -> Tensor:
        """
        Finalize token reconstruction with unpermutation and reshaping.

        This method unpermutes the tokens back to their original order,
        reshapes the tensor to its original shape, and adds the shared expert 
        output if enabled.

        Args:
            permutated_local_input_tokens: permuted hidden states from token combine.

        Returns:
            the final MoE layer output reshaped to its original dimensions.
        """
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
            self.shared_experts.post_forward_comm()

        # Unpermutation 1: AlltoAll output to output
        output = permute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            drop_and_pad=self.drop_and_pad
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        # Add shared expert output
        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts.get_output()
            output += shared_expert_output
        return output
    
    def _maybe_update_cuda_sync_point(self, point: str):
        """
        Update the cuda sync point if the priority of the new point is higher than the current
        sync point, which means the new point is reached earlier than the current sync point.
        """
        if (
            self.cuda_sync_point_priority[point]
            < self.cuda_sync_point_priority[self.cuda_sync_point]
        ):
            self.cuda_sync_point = point

    def _maybe_d2h_and_synchronize(
        self, point: str, tokens_per_expert: Optional[Tensor] = None
    ) -> Tensor:
        """
        Move all possible device tensors to host and make a synchronization at the expected point.
        """
        if not self.drop_and_pad:
            if point == self.cuda_sync_point:
                # Move all possible device tensors to host at self.cuda_d2h_point.
                on_side_stream = torch.cuda.Stream() != self.d2h_stream
                if on_side_stream:
                    self.d2h_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.d2h_stream):
                    tokens_per_expert = maybe_move_tensor_to_cpu(
                        tokens_per_expert, record_stream=on_side_stream
                    )
                    self.input_splits = maybe_move_tensor_to_cpu(
                        self.input_splits, record_stream=on_side_stream
                    )
                    self.output_splits = maybe_move_tensor_to_cpu(
                        self.output_splits, record_stream=on_side_stream
                    )
                    self.output_splits_tp = maybe_move_tensor_to_cpu(
                        self.output_splits_tp, record_stream=on_side_stream
                    )
                    self.num_out_tokens = maybe_move_tensor_to_cpu(
                        self.num_out_tokens, record_stream=on_side_stream
                    )
                    if self.num_local_experts > 1 and not self.config.moe_permute_fusion:
                        self.num_global_tokens_per_local_expert = maybe_move_tensor_to_cpu(
                            self.num_global_tokens_per_local_expert, record_stream=on_side_stream
                        )
                self.d2h_event = self.d2h_stream.record_event()

            if point == self.cuda_sync_point:
                # Synchronize with the d2h stream at self.cuda_sync_point.
                self.d2h_event.synchronize()
            
        return tokens_per_expert
    