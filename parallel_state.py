import logging
from typing import Optional, List, Callable
from math import log2
import einops

import torch

from moe_utils import GlobalMemoryBuffer


# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group(both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Context parallel group that the current rank belongs to.
_CONTEXT_PARALLEL_GROUP = None
# Expert parallel group that the current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
# Expert tensor parallel group that the current rank belongs to.
_EXPERT_TENSOR_PARALLEL_GROUP = None
# Tensor and expert parallel group that the current rank belongs to.
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Expert data parallel group that the current rank belongs to.
_EXPERT_DATA_PARALLEL_GROUP = None

# _EXPERT_MODEL denotes expert parallelism which splits the number of experts across the group
# _EXPERT_TENSOR denotes tensor parallelism of expert which parameters across the group
# _EXPERT_DATA denotes data parallelism of expert which replicates weight across the group

