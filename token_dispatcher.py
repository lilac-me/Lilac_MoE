from typing import Optional
from abc import abstractmethod

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
        self.shared_expert = Optional[SharedExpertMLP] = None

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
        