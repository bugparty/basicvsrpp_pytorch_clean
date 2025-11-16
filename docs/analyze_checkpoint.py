#!/usr/bin/env python3
"""
分析官方 checkpoint 与我们模型的参数差异
"""

import sys
import torch
from collections import defaultdict

sys.path.insert(0, '/home/bowman/basicvsrpp_pytorch_clean')

from basicvsrpp.inference import get_default_gan_inference_config
from basicvsrpp.mmagic.registry import MODELS
from basicvsrpp import register_all_modules


def analyze_checkpoint(checkpoint_path):
    """分析 checkpoint 的参数"""

    print(f"加载 checkpoint: {checkpoint_path}\n")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 获取 state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 统计参数
    print("=" * 80)
    print("Checkpoint 参数统计")
    print("=" * 80)

    # 按模块分类
    modules = defaultdict(list)
    for key in state_dict.keys():
        if '.' in key:
            module = key.split('.')[0]
        else:
            module = 'root'
        modules[module].append(key)

    print(f"\n总参数数量: {len(state_dict)}\n")

    for module, keys in sorted(modules.items()):
        print(f"{module:30s}: {len(keys):4d} 个参数")

    # 分析 generator 和 generator_ema
    print("\n" + "=" * 80)
    print("Generator 结构分析")
    print("=" * 80)

    gen_keys = [k for k in state_dict.keys() if k.startswith('generator.')]
    gen_ema_keys = [k for k in state_dict.keys() if k.startswith('generator_ema.')]

    print(f"\ngenerator (非EMA版本):     {len(gen_keys):4d} 个参数")
    print(f"generator_ema (EMA版本):    {len(gen_ema_keys):4d} 个参数")

    # 分析 generator_ema 的子模块
    if gen_ema_keys:
        print("\ngenerator_ema 子模块:")
        gen_ema_modules = defaultdict(list)
        for key in gen_ema_keys:
            parts = key.split('.')
            if len(parts) >= 2:
                submodule = parts[1]
                gen_ema_modules[submodule].append(key)

        for submodule, keys in sorted(gen_ema_modules.items()):
            print(f"  {submodule:30s}: {len(keys):4d} 个参数")

    # 分析 backbone ResBlocks 数量
    print("\n" + "=" * 80)
    print("Backbone ResBlocks 分析")
    print("=" * 80)

    for prefix in ['generator.', 'generator_ema.']:
        for direction in ['backward_1', 'forward_1', 'backward_2', 'forward_2']:
            pattern = f'{prefix}backbone.{direction}.main.2.'
            blocks = set()
            for key in state_dict.keys():
                if key.startswith(pattern):
                    # 提取 block 索引
                    rest = key[len(pattern):]
                    if rest and rest[0].isdigit():
                        block_idx = rest.split('.')[0]
                        blocks.add(int(block_idx))

            if blocks:
                max_block = max(blocks)
                print(f"{prefix}{direction:15s}: {max_block + 1} 个 ResBlocks (索引 0-{max_block})")

    return state_dict


def compare_with_model(checkpoint_path):
    """对比 checkpoint 与模型结构"""

    print("\n" + "=" * 80)
    print("对比 Checkpoint 与模型结构")
    print("=" * 80)

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        ckpt_state_dict = checkpoint['state_dict']
    else:
        ckpt_state_dict = checkpoint

    # 创建模型
    register_all_modules()
    config = get_default_gan_inference_config()
    model = MODELS.build(config)
    model_state_dict = model.state_dict()

    # 找出差异
    ckpt_keys = set(ckpt_state_dict.keys())
    model_keys = set(model_state_dict.keys())

    unexpected = ckpt_keys - model_keys  # checkpoint 有但模型没有
    missing = model_keys - ckpt_keys     # 模型需要但 checkpoint 没有
    matched = ckpt_keys & model_keys     # 匹配的参数

    print(f"\n匹配的参数:     {len(matched):4d}")
    print(f"Unexpected keys: {len(unexpected):4d} (checkpoint有但模型不需要)")
    print(f"Missing keys:    {len(missing):4d} (模型需要但checkpoint没有)")

    # 分类 unexpected keys
    if unexpected:
        print("\n--- Unexpected Keys 分类 ---")
        unexpected_modules = defaultdict(int)
        for key in unexpected:
            module = key.split('.')[0] + '.' + key.split('.')[1] if '.' in key else key
            unexpected_modules[module] += 1

        for module, count in sorted(unexpected_modules.items(), key=lambda x: -x[1])[:10]:
            print(f"  {module:40s}: {count:4d} 个参数")

        if len(unexpected_modules) > 10:
            print(f"  ... 还有 {len(unexpected_modules) - 10} 个模块")

    # 分类 missing keys
    if missing:
        print("\n--- Missing Keys 分类 ---")
        missing_modules = defaultdict(int)
        for key in missing:
            module = key.split('.')[0] + '.' + key.split('.')[1] if '.' in key else key
            missing_modules[module] += 1

        for module, count in sorted(missing_modules.items(), key=lambda x: -x[1])[:10]:
            print(f"  {module:40s}: {count:4d} 个参数")

        if len(missing_modules) > 10:
            print(f"  ... 还有 {len(missing_modules) - 10} 个模块")

    # 检查关键模块是否匹配
    print("\n" + "=" * 80)
    print("关键模块匹配情况")
    print("=" * 80)

    critical_modules = [
        'generator_ema.spynet',
        'generator_ema.feat_extract',
        'generator_ema.deform_align',
        'generator_ema.backbone',
        'generator_ema.reconstruction',
    ]

    for module in critical_modules:
        module_matched = len([k for k in matched if k.startswith(module + '.')])
        module_missing = len([k for k in missing if k.startswith(module + '.')])
        module_total = module_matched + module_missing

        if module_total > 0:
            match_rate = module_matched / module_total * 100
            status = "✅" if match_rate > 90 else "⚠️" if match_rate > 50 else "❌"
            print(f"{status} {module:35s}: {match_rate:5.1f}% ({module_matched}/{module_total})")


if __name__ == '__main__':
    checkpoint_path = '/home/bowman/basicvsrpp_pytorch_clean/checkpoints/basicvsr_plusplus_reds4.pth'

    # 分析 checkpoint
    analyze_checkpoint(checkpoint_path)

    # 对比模型
    compare_with_model(checkpoint_path)

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    print("\n详细文档请查看: docs/CHECKPOINT_COMPATIBILITY.md")
