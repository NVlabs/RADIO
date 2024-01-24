from collections import defaultdict
from contextlib import contextmanager
import math
from typing import List, Dict, Union, Tuple, Optional

import torch
import torch.distributed as dist

from .resize_transform import ResizeTransform


def round_up(value, multiple: int):
    return int(math.ceil(value / multiple))


def collate(samples: List[Dict[str, torch.Tensor]]):
    images = [
        s['image']
        for s in samples
    ]
    labels = [
        s['label']
        for s in samples
    ]

    size_groups = defaultdict(lambda: [[],[]])
    for im, lab in zip(images, labels):
        grp = size_groups[im.shape]
        grp[0].append(im)
        grp[1].append(lab)

    ret = [
        (torch.stack(g[0]), torch.stack(g[1]))
        for g in size_groups.values()
    ]
    return ret


@contextmanager
def run_rank_0_first(group: Optional[dist.ProcessGroup] = None):
    rank = get_rank(group)
    world_size = get_world_size(group)

    if rank == 0:
        yield

    if world_size > 1:
        # Rank 0 will process itself first, whereas all other ranks will hit the barrier first
        dist.barrier(group)

        if rank > 0:
            yield


def get_rank(group: Optional[dist.ProcessGroup] = None):
    return dist.get_rank(group) if dist.is_initialized() else 0


def get_world_size(group: Optional[dist.ProcessGroup] = None):
    return dist.get_world_size(group) if dist.is_initialized() else 1
