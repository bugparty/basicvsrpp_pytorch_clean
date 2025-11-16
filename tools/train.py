#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Training script for BasicVSR++.

Example:
    Single GPU:
        python tools/train.py configs/train_basicvsrpp.py

    Multiple GPUs:
        torchrun --nproc_per_node=4 tools/train.py configs/train_basicvsrpp.py

    Resume from checkpoint:
        python tools/train.py configs/train_basicvsrpp.py --resume work_dirs/train/latest.pth
"""

import argparse
import os
import os.path as osp
import sys

# Add project root to path
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from basicvsrpp import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='resume training from a checkpoint. If "auto", resume from the latest checkpoint in work-dir')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic mixed precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR based on batch size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Register all modules
    register_all_modules()

    # Load config
    cfg = Config.fromfile(args.config)

    # Merge CLI arguments into config
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # Use config filename as default work_dir if not specified
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # Enable automatic mixed precision training
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print('AMP training is already enabled in your config.')
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # Disable validation if specified
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    # Resume training
    if args.resume:
        cfg.resume = True
        if args.resume != 'auto':
            cfg.load_from = args.resume

    # Auto scale learning rate
    if args.auto_scale_lr:
        # Get the base batch size from config
        base_batch_size = cfg.get('base_batch_size', None)
        if base_batch_size is None:
            raise ValueError(
                'Please set `base_batch_size` in your config to use '
                '`--auto-scale-lr`')

        # Calculate current total batch size
        samples_per_gpu = cfg.train_dataloader.batch_size
        num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
        total_batch_size = samples_per_gpu * num_gpus

        # Scale learning rate
        scale_ratio = total_batch_size / base_batch_size

        if 'optim_wrapper' in cfg:
            if 'generator' in cfg.optim_wrapper:
                cfg.optim_wrapper.generator.optimizer.lr *= scale_ratio
            if 'discriminator' in cfg.optim_wrapper:
                cfg.optim_wrapper.discriminator.optimizer.lr *= scale_ratio

        print(f'Auto-scaling learning rate by {scale_ratio:.2f} '
              f'(batch size: {base_batch_size} -> {total_batch_size})')

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()


if __name__ == '__main__':
    main()
