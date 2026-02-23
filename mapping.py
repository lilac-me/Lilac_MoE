from typing import Any
import torch


def _gather_along_first_dim(input_, group, output_split_sizes=None, use_global_buffer=False):
    """
    Gather tensors and concatenate along the first dimension
    
    Args:
        input_: the tensor to be gathered.
        output_split_sizes: List[int], a list specifying the sizes of the output splits along the 
                            first dimension. If None, equal splitting is assumed. Default to None.
    """
    assert group is not None, "group should not be None"
    world_size = group.size()
    if world_size == 1:
        return input_
    
    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size
        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        dist_all_gather_func(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def gather_from_sequence_parallel_region(
    input_,
    tensor_parallal_output_grad=True,
    group=None,
    output_split_sizes=None,
    use_golbal_buffer=False,
):
    """
    Wrapper for autograd function: forward: AG, backward: RS <first_dim>.
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _GatherFromSequenceParallelRegion.apply(
        input_, group, tensor_parallal_output_grad, output_split_sizes, use_golbal_buffer
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
        otuput_split_sizes=None,
        use_global_buffer=False,
    ):
        """
        Symbolic function for tracing.
        """
        return _gather_along_first_dim(input_, group, otuput_split_sizes, use_global_buffer)
    
    @staticmethod
    def forward(
        ctx,
        input_,
        group,
        tensor_parallel_output_grad=True,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        ctx.group = group
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.output_split_sizes = output_split_sizes
        ctx.use_global_buffer = use_global_buffer
        return _gather_along_first_dim(input_, group, output_split_sizes, use_global_buffer)
    
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
                    grad_output, ctx.group, ctx.output_split_sizes, ctx.use_global_buffer
                ),
                None,
                None,
                None,
                None
            )
        else:
            assert ctx.output_split_sizes is not None
            return (_split_along_first_dim(grad_output, ctx.group), None, None, None, None)
