import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import parallel_state
from process_group_collection import ProcessGroupCollection
from transformer_config import TransformerConfig
from token_dispatcher import MoEAllgatherTokenDispatcher, MoEAlltoAllTokenDispatcher


# A simple expert computation for simulation.
class FakeExpert(torch.nn.Module):
    def __init__(self, hidden_size, moe_ffn_size):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_size, moe_ffn_size, bias=False)
    
    def forward(self, x):
        return self.fc(x)
    

def run(args):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    # torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="gloo")

    tp_size = 1
    ep_size = world_size
    num_experts = world_size * 2
    num_local_experts = 2
    local_expert_indices = list(range(2 * rank, 2 * (rank + 1))) # [0, 1], [2, 3], [4, 5], [6, 7]

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
        order="tp-ep-dp-pp",
        create_gloo_process_groups=True,
    )

    pg_collection = ProcessGroupCollection.use_mpu_process_group(
        required_pgs=["tp", "ep", "expt_tp", "tp_ep", "dp", "expt_dp", "pp", "mp", "cp"]
    )

    config = TransformerConfig(
        num_moe_experts=num_experts,
        moe_router_topk=2,
        hidden_size=16,
        add_bias_linear=False,
    )

    if args.dispatcher_type == "alltoall":
        dispatcher = MoEAlltoAllTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=pg_collection,
        )
    else:
        dispatcher = MoEAllgatherTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=pg_collection,
        )

    hidden_size = config.hidden_size
    moe_ffn_size = config.moe_intermediate_size
    expert = FakeExpert(hidden_size, moe_ffn_size)

    S, B, H = 16, 1, hidden_size
    hidden_states = torch.randn(S, B, H)

    num_tokens = S * B
    topk = config.moe_router_topk
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    probs = torch.zeros(num_tokens, num_experts)

    for i in range(num_tokens):
        chosen = torch.randperm(num_experts)[:topk]
        routing_map[i, chosen] = True
        probs[i, chosen] = torch.softmax(torch.randn(topk), dim=0)

    # dispatch preprocess
    permuted_local_hidden_states, permuted_local_probs = dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)

    # dispatch
    dispatch_hidden_states, dispatch_probs = dispatcher.token_dispatch(permuted_local_hidden_states, permuted_local_probs)

    # dispatch postprocess
    permuted_tokens, token_per_expert, permuted_probs = dispatcher.dispatch_postprocess(dispatch_hidden_states, dispatch_probs)

    # expert computation
    expert_output = expert(permuted_tokens)

    # combine preprocess
    unpermuted_local_hidden_states = dispatcher.combine_preprocess(expert_output)

    # combine
    combine_hidden_states = dispatcher.token_combine(unpermuted_local_hidden_states)

    # combine postprocess
    output = dispatcher.combine_postprocess(combine_hidden_states)

    if rank == 0:
        print("Master Rank: Dispatch and Combine finished.")
    print(
        f"[rank {rank}] input: {hidden_states.shape}"
        f"tokens_per_expert: {token_per_expert.tolist()}"
        f"output: {output.shape}"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lilac MoE simulation.")
    parser.add_argument(
        "--dispatcher-type",
        type=str,
        default="alltoall",
        choices=["alltoall", "allgather"],
        help="Type of MoE token dispatcher"
    )

    args = parser.parse_args()

    run(args)
