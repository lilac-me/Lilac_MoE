from dataclasses import dataclass, field, fields
from typing import Optional, List
from functools import partial

import torch

import parallel_state

@dataclass
class ProcessGroupCollection:
    """
    Unified process group collection for transformer model parallelism, gradient accumulation
    and finalization.
    Fields use init=False and must be set after instance creation.

    Args:
        tp: tensor parallel process group
        pp: pipeline parallel process group
        mp: model parallel process group
        cp: context parallel process group
        ep: expert parallel process group
        expt_tp: expert tensor parallel process group (etp)
        tp_ep: tensor and expert parallel process group

        dp: data parallel process group
        expt_dp: expert data parallel process group (edp)
    
    Example:
        pgs = ProgressGroupCollection()
        pgs.tp = tp_group
        pgs.pp = pp_group
        pgs.dp = dp_group

        model = TransformerModel(..., pg_collection=pgs)
        ddp_model = DistributedDataParallel(..., pg_collection=pgs)
        finalize_model_grads(..., pg_collection=pgs)
    """
    # Model Parallel Process Groups
    # _TENSOR_MODEL_PARALLEL_GROUP
    tp: torch.distributed.ProcessGroup = field(init=False)

    # _PIPELINE_MODEL_PARALLEL_GROUP
    pp: torch.distributed.ProcessGroup = field(init=False)

    # _MODEL_PARALLEL_GROUP
    mp: torch.distributed.ProcessGroup = field(init=False)

    # _CONTEXT_PARALLEL_GROUP
    cp: torch.distributed.ProcessGroup = field(init=False)

    # _EXPERT_MODEL_PARALLEL_GROUP
    ep: torch.distributed.ProcessGroup = field(init=False)

    # _EXPERT_TENSOR_PARALLEL_GROUP
    expt_tp: torch.distributed.ProcessGroup = field(init=False)

    # _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    tp_ep: torch.distributed.ProcessGroup = field(init=False)

    # _DATA_PARALLEL_GROUP
    dp: torch.distributed.ProcessGroup = field(init=False)

    # MoE layers need edp for shared state dict
    # _EXPERT_DATA_PARALLEL_GROUP
    expt_dp: torch.distributed.ProcessGroup = field(init=False)

    def __init__(self, **kwargs):
        for key in kwargs:
            if key in [field.name for field in fields(self)]:
                setattr(self, key, kwargs[key])
            else:
                raise ValueError(f"Invalid field name {key} for ProgressGroupCollection")
            
    def __repr__(self) -> str:
        """
        Return a concise representation showing which process groups exists and their sizes.
        """
        active_pgs = []
        for field_info in fields(self):
            if hasattr(self, field_info.name):
                pg = getattr(self, field_info.name)
                if pg is not None:
                    active_pgs.append(f"{field_info.name}({pg.size()})")
                else:
                    active_pgs.append(f"{field_info.name}(None)")
        return (
            f"ProgressGroupCollection({', '.join(active_pgs)})"
            if active_pgs
            else "ProgressGroupCollection(empty)"
        )
    
    @classmethod
    def use_mpu_process_group(cls, required_pgs: Optional[List[str]] = None):
        """
        Use the default process groups from parallel_state.

        Args:
            required_pgs: A list of required process group names to initialize.
        """
        # get all available process groups
        all_pgs = {field.name for field in fields(cls)}

        # if no specific process group requested, use all
        if required_pgs is None:
            required_pgs = list(all_pgs)

        # validate requested process groups
        invalid_pgs = [pg for pg in required_pgs if pg not in all_pgs]
        if invalid_pgs:
            raise ValueError(f"Invalid process group names: {invalid_pgs}.")
        
        # mapping of arribute names to parallel_state functions
        pgs_to_func = {
            'tp': partial(parallel_state.get_tensor_model_parallel_group, check_initialized=False),
            'pp': partial(parallel_state.get_pipeline_model_parallel_group, check_initialized=False),
            'mp': partial(parallel_state.get_model_parallel_group, check_initialized=False),
            'cp': partial(parallel_state.get_context_parallel_group, check_initialized=False),
            'ep': partial(parallel_state.get_expert_model_parallel_group, check_initialized=False),
            'expt_tp': partial(parallel_state.get_expert_tensor_parallel_group, check_initialized=False),
            'tp_ep': partial(parallel_state.get_expert_tensor_and_model_parallel_group, check_initialized=False),
            'dp': partial(parallel_state.get_data_parallel_group, check_initialized=False),
            'expt_dp': partial(parallel_state.get_expert_data_parallel_group, check_initialized=False)
        }

        assert all(
            pg in pgs_to_func for pg in required_pgs
        ), f"Some required process groups are not available: {required_pgs}"

        init_dict = {pg: pgs_to_func[pg]() for pg in required_pgs}

        return cls(**init_dict)
    