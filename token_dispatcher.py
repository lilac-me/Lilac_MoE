from typing import Optional, List
from abc import abstractmethod

import torch
from torch import Tensor
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
            routing_map (Tensor): 2D [S/TP*B, num_experts].
            probs (Tensor): 2D [S/TP*B, num_experts].
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
        After gatherting these  tokens, this method identifies the tokens for local experts and
        permutes the tokens to group them by expert for efficient expert processing.

        Args:
            hidden_states (Tensor): 2D [S*B*EP, H].
            probs (Tensor): 2D [S*B*EP, num_experts].
        """
        self.hidden_shape_before_permute = hidden_states.shape

        # The routing map and probs that for local experts. [S*B*EP, num_local_experts]
        self.local_map = self.routing_map[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # Probs of global token assignment to local experts. [S*B*EP, num_local_experts]
        self.local_probs = probs[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()

        tokens_per_expert = self.local_map.sum(dim=0).long().cpu() # [num_local_experts]

        (permuted_local_hidden_states, _, self.reversed_local_input_permutaiton_mapping) = permute(
            hidden_states, # [S*B*EP, H]
            self.local_map, # [S*B*EP, num_local_experts]
            num_out_tokens=tokens_per_expert.sum().item(),
            fused=self.config.moe_permute_fusion,
        )
        # permuted_local_hidden_states: [num_local_tokens, H]
        # self.reversed_local_input_permutaiton_mapping: [num_local_tokens]

        self.local_probs = self.local_probs.T.contiguous().masked_select(
            self.local_map.T.contiguous()
        )
        # self.local_probs.T: [num_local_tokens, S*B*EP]
        # self.local_map.T: [num_local_tokens, S*B*EP]
        self.routing_map = None
        return permuted_local_hidden_states, tokens_per_expert, self.local_probs
    
    def combine_postprocess(self, hidden_states: Tensor) -> Tensor:
        """
        Reverses token permutation to restore original ordering before reduction operations.

        This method unpermutes the expert outputs using the cached permutation mapping
        from the dispatch phase. The unpermutation operation restores tokens to their
        original sequence positions, preparing them for the subsequent reduction scatter
        operation that will aggregate results across ranks.
        """
        unpermuted_local_hidden_states = unpermute(
            hidden_states,
            self.reversed_local_input_permutaiton_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.local_map,
            fused=self.config.moe_permute_fusion,
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
    
    def combine_preprocess(self, hidden_states: Tensor) -> Tensor:
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
        