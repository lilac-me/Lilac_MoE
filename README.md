# Lilac_MoE

A standalone, educational implementation of Megatron-style **Mixture-of-Experts (MoE) token dispatching**, extracted and refactored from [Megatron-Core](https://github.com/NVIDIA/Megatron-LM). Covers both the **AllGather** and **AlltoAll** communication paradigms for Expert Parallelism (EP).

## Why This Repo?

Megatron-LM's MoE implementation is deeply intertwined with its training framework, making it difficult to study the dispatch logic in isolation. This project pulls out the core token-routing and communication primitives so that:

- You can **read and understand** how MoE dispatch/undispatch actually works without wading through thousands of lines of training scaffolding.
- You can **experiment** with different EP strategies (AllGather vs AlltoAll) and observe the trade-offs.
- You can **reuse** the dispatch primitives in your own training loop or framework.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                       Router / Gate                       │
│           routing_map: [num_tokens, num_experts]          │
└───────────────────────────┬──────────────────────────────┘
                            │
                ┌───────────▼────────────┐
                │    Token Dispatcher     │  ← token_dispatcher.py
                │  ┌───────────────────┐  │
                │  │   AllGather Mode  │  │  Replicate tokens → local experts
                │  │   AlltoAll Mode   │  │  Shard tokens → remote experts
                │  └───────────────────┘  │
                └───────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │  Expert Parallel Group   │  ← parallel_state.py
               │  Process Group Mgmt      │  ← process_group_collection.py
               └─────────────────────────┘
```

### AllGather vs AlltoAll — When to Use Which?

| | **AllGather** | **AlltoAll** |
|---|---|---|
| **Communication** | Each EP rank gathers all tokens, then computes on its local experts | Tokens are exchanged between EP ranks so each rank only processes tokens routed to its experts |
| **Memory** | Higher — every rank holds a full copy of tokens | Lower — each rank only holds tokens destined for its experts |
| **Compute** | May do redundant work if capacity factor is low | Minimal redundant work |
| **Best for** | Small EP sizes, small token counts | Large EP sizes, large-scale training |
| **CUDA Graph** | Easier (static shapes with drop-and-pad) | Harder (dynamic shapes) |

## File Structure

```
Lilac_MoE/
├── token_dispatcher.py          # Core: AllGather & AlltoAll dispatcher implementations
│                                #   - MoEAllGatherTokenDispatcher
│                                #   - MoEAlltoAllTokenDispatcher
│                                #   - dispatch() / undispatch() entry points
│
├── moe_utils.py                 # Token permutation, unpermutation, capacity calculation,
│                                #   global memory buffer for zero-allocation dispatch
│
├── transformer_config.py        # MoE-related config dataclass (num_experts, capacity_factor,
│                                #   top_k, expert_parallel_size, etc.)
│
├── parallel_state.py            # Expert-parallel group initialization and rank queries
│
├── process_group_collection.py  # ProcessGroupCollection — abstracts TP/EP/DP group handles
│
├── mapping.py                   # Rank mapping utilities for multi-dimensional parallelism
│
├── utils.py                     # Miscellaneous helpers
│
└── README.md
```

## Key Concepts

### Token Dispatch (Forward)

```
Input:  hidden_states  [S, B, H]    (sequence, batch, hidden)
        routing_map    [S*B, E]     (token-to-expert assignment, sparse bool)
        probs          [S*B, E]     (gating scores)

  ┌─────────────┐     permute by      ┌──────────────────┐
  │ hidden_states│ ──────────────────► │ permuted_tokens   │  (grouped by expert)
  └─────────────┘   routing_map       └────────┬─────────┘
                                               │
                                    AllGather or AlltoAll
                                               │
                                      ┌────────▼─────────┐
                                      │  dispatched tokens │  (ready for expert FFN)
                                      └──────────────────┘
```

### Token Undispatch (Backward-compatible reverse)

```
  expert_output  ──► AlltoAll/AllGather reverse ──► unpermute ──► weighted combine ──► output
```

### Drop-and-Pad

When `drop_and_pad=True`, each expert processes exactly `capacity` tokens. Excess tokens are dropped; deficit is zero-padded. This makes tensor shapes static and enables **CUDA graph capture**.

### Load Balancing Loss

The auxiliary load-balancing loss encourages uniform expert utilization, calculated from the routing probabilities and the actual token assignments. Controlled by `moe_aux_loss_coeff` in the config.

## Dependencies

- Python >= 3.8
- PyTorch >= 2.0 (with CUDA support for multi-GPU)
- NCCL (for distributed communication)

No dependency on Megatron-LM itself — this repo is self-contained.

## Quick Start

```python
import torch
from transformer_config import TransformerConfig
from token_dispatcher import MoEAllGatherTokenDispatcher

# Configure
config = TransformerConfig(
    num_moe_experts=8,
    moe_router_topk=2,
    moe_expert_capacity_factor=1.5,
    expert_model_parallel_size=1,   # single-GPU for demo
)

# Create dispatcher
dispatcher = MoEAllGatherTokenDispatcher(
    num_local_experts=8,
    local_expert_indices=list(range(8)),
    config=config,
)

# Simulate router output
num_tokens = 64
hidden_size = 256
probs = torch.randn(num_tokens, 8).softmax(dim=-1).cuda()
routing_map = torch.zeros(num_tokens, 8, dtype=torch.bool).cuda()
# Top-2 routing
topk_indices = probs.topk(2, dim=-1).indices
for i in range(num_tokens):
    for j in topk_indices[i]:
        routing_map[i, j] = True

# Dispatch
hidden_states = torch.randn(num_tokens, hidden_size).cuda()
dispatched, tokens_per_expert = dispatcher.dispatch(hidden_states, probs, routing_map)

# ... run through expert FFNs ...
expert_output = dispatched  # placeholder

# Undispatch
output, bias = dispatcher.undispatch(expert_output)
# output shape: [num_tokens, hidden_size]
```

> **Note**: For multi-GPU AlltoAll, you need to initialize the EP process groups via `parallel_state.py` before creating the dispatcher.

