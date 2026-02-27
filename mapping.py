from typing import Any
import torch

from moe_utils import get_global_memory_buffer


device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"


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
        output = torch.empty(dim_size, dtype=input_.dtype, device=device)
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
        output = torch.empty(dim_size, dtype=input_.dtype, device=device)
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
    tensor_parallel_output_grad=True,
    group=None,
    use_global_buffer=False,
):
    """
    Wrapper for autograd function: forward: AG, backward: RS <first_dim>.
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _GatherFromSequenceParallelRegion.apply(
        input_, group, tensor_parallel_output_grad, use_global_buffer
    )


def reduce_scatter_to_sequence_parallel_region(
    input_,
    group=None,
    use_global_buffer=False,
):
    """
    Wrapper for autograd function: forward: RS, backward: AG <first_dim>.
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _ReduceScatterToSequenceParallelRegion.apply(
        input_, group, use_global_buffer
    )


def all_to_all(group, input_, output_split_sizes=None, input_split_sizes=None):
    assert group is not None, "group should not be None"
    _AllToAll.apply(group, input_, output_split_sizes, input_split_sizes)


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


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """
    Reduce-Scatter the input from the sequence parallel region.
    """
    @staticmethod
    def symbolic(
        graph,
        input_,
        group,
        use_global_buffer=False,
    ):
        """
        Symbolic function for tracing.
        """
        return _reduce_scatter_along_first_dim(input_, group, use_global_buffer)
    
    @staticmethod
    def forward(ctx, input_, group, use_global_buffer=False):
        ctx.group = group
        ctx.use_global_buffer = use_global_buffer
        return _reduce_scatter_along_first_dim(input_, group, use_global_buffer)
    
    @staticmethod
    def backward(ctx, grad_output):
        use_global_buffer = ctx.use_global_buffer
        return (
            _gather_along_first_dim(grad_output, ctx.group, use_global_buffer),
            None,
            None,
            None,
        )
    

class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input_, output_split_sizes=None, input_split_sizes=None):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = group.size()
        if world_size == 1:
            return input_
        input_ = input_.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input_)
        else:
            # Unequal split (all2all-v)
            output = torch.empty(
                size=[sum(output_split_sizes)] + list(input_.size()[1:]),
                dtype=input_.dtype,
                device=input_.device,
            )
        torch.distributed.all_to_all_single(
            output,
            input_,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        return (
            None,
            _AllToAll.apply(
                ctx.group,
                *grad_outputs,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
            ),
            None,
            None,
        )
    