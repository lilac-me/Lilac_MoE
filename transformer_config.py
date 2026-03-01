import torch
import dataclasses

@dataclasses.dataclass
class TransformerConfig:
    num_moe_experts: int = 1
    moe_router_topk: int = 1
    moe_shared_expert_overlap: bool = False
    moe_pad_expert_input_to_capacity: bool = False
    moe_expert_capacity_factor: float = 1.0
    moe_router_padding_for_quantization: bool = False
    add_bias_linear: bool = False
