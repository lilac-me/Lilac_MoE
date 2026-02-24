from typing import Any
import torch


def _gather_along_first_dim(input_, group, use_global_buffer=False):
    """
    Gather tensors and concatenate along the first dimension.
    equal splitting is assumed.
    
    Args:
        input_: the tensor to be gathered.
    """
    assert group is not None, "group should not be None"
    world_size = group.size()
    if world_size == 1:
        return input_
    
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    if use_global_buffer:
        output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
    else:
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)

    return output


def _reduce_scatter_along_first_dim(input_, group, use_global_buffer=False):
    """
    Reduce-Scatter the input tensor across model parallel group.
    equal splitting is assumed.

    Args:
        input_: the input tensor to be reduce-scattered
    """
    assert group is not None, "group should not be None"
    world_size = group.size()
    if world_size == 1:
        return input_
    
    dim_size = list(input_.size())
    assert(
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size."
    dim_size[0] = dim_size[0] // world_size

    if use_global_buffer:
        output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
    else:
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.reduce_scatter_tensor(output, input_.contiguous(), group=group)
    
    return output


def _split_along_first_dim(input_, group):
    """
    Split the tensor along its first dimension and keep the correponding slices.
    """
    assert group is not None, "group should not be None"
    world_size = group.size()
    if world_size == 1:
        return input_
    
    dim_size = input_.size(0)
    assert(
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size."
    local_dim_size = dim_size // world_size
    rank = group.rank()
    dim_offset = rank * local_dim_size
    
    output = input_[dim_offset: dim_offset + local_dim_size].contiguous()

    return output


def gather_from_sequence_parallel_region(
    input_,
    tensor_parallal_output_grad=True,
    group=None,
    use_golbal_buffer=False,
):
    """
    Wrapper for autograd function: forward: AG, backward: RS <first_dim>.
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _GatherFromSequenceParallelRegion.apply(
        input_, group, tensor_parallal_output_grad, use_golbal_buffer
    )


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """
    Gather the input from sequence parallel region and concatenate.
    """
    @staticmethod
    def symbolic(
        graph,
        input_,
        group,
        tensor_parallel_output_grad=True,
        use_global_buffer=False,
    ):
        """
        Symbolic function for tracing.
        """
        return _gather_along_first_dim(input_, group, use_global_buffer)
    
    @staticmethod
    def forward(
        ctx,
        input_,
        group,
        tensor_parallel_output_grad=True,
        use_global_buffer=False,
    ):
        ctx.group = group
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.use_global_buffer = use_global_buffer
        return _gather_along_first_dim(input_, group, use_global_buffer)
    
    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is 
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return (
                _reduce_scatter_along_first_dim(
                    grad_output, ctx.group, ctx.use_global_buffer
                ),
                None,
                None,
                None,
                None
            )
        else:
            return (_split_along_first_dim(grad_output, ctx.group), None, None, None, None)
