import torch
import torch.distributed as dist

def all_to_all_single_test():
    dist.init_process_group(backend='gloo')
    world_size = dist.get_world_size() # --nproc_per_node=4 means world_size=4
    rank = dist.get_rank() # rank is the unique identifier of each process, ranging from 0 to world_size-1

    # Create a tensor with the rank of the process
    send_tensor = torch.full((world_size,), rank)
    recv_tensor = torch.empty_like(send_tensor)

    print(f"Rank {rank} send tensor:\n {send_tensor} \n")

    # Perform all-to-all communication
    dist.all_to_all_single(recv_tensor, send_tensor)

    print(f"Rank {rank} received tensor:\n {recv_tensor} \n")
    dist.destroy_process_group()

if __name__ == "__main__":
    all_to_all_single_test()