# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import hashlib
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS

# Wildcard imports are considered bad practice but in this instance they
# serve as a means for registering modules and setting up appropriate sys paths.
from linear_head import *
from radio import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentor")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--work-dir", "--output", help="the dir to save logs and models"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from the latest checkpoint in the work_dir automatically",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic-mixed-precision training",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    for vis_backend in cfg.visualizer.vis_backends:
        if vis_backend.type == "WandbVisBackend":
            # Name the job type after config file name.
            job_type = osp.splitext(osp.basename(args.config))[0]
            # Name the job after the last part of the work directory.
            job_name = cfg.work_dir.split("/")[-1]
            # Hash job name into a 8-digit id.
            job_id = hashlib.sha256(job_name.encode()).hexdigest()[:8]
            vis_backend.init_kwargs.job_type = job_type
            vis_backend.init_kwargs.name = job_name
            vis_backend.init_kwargs.id = job_id
            vis_backend.init_kwargs.resume = "allow"
            vis_backend.init_kwargs.allow_val_change = True

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == "AmpOptimWrapper":
            print_log(
                "AMP training is already enabled in your config.",
                logger="current",
                level=logging.WARNING,
            )
        else:
            assert optim_wrapper == "OptimWrapper", (
                "`--amp` is only supported when the optimizer wrapper type is "
                f"`OptimWrapper` but got {optim_wrapper}."
            )
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == "__main__":
    main()
