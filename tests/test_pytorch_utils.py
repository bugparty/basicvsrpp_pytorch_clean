"""
测试 pytorch_utils.py 中的工具函数

运行方式:
    python tests/test_pytorch_utils.py
    或
    pytest tests/test_pytorch_utils.py -v
"""

import torch
import torch.nn as nn
import tempfile
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from basicvsrpp.mmagic.pytorch_utils import (
    constant_init,
    kaiming_init,
    xavier_init,
    normal_init,
    default_init_weights,
    load_checkpoint,
    save_checkpoint,
    get_logger,
    get_module_device,
    set_requires_grad,
    is_batch_norm,
)


def test_constant_init():
    """测试常量初始化"""
    print("Testing constant_init...")

    conv = nn.Conv2d(3, 64, 3)
    constant_init(conv, val=1.0, bias=0.5)

    assert torch.all(conv.weight == 1.0), "Weight should be 1.0"
    assert torch.all(conv.bias == 0.5), "Bias should be 0.5"

    print("✅ constant_init passed")


def test_kaiming_init():
    """测试 Kaiming 初始化"""
    print("Testing kaiming_init...")

    conv = nn.Conv2d(3, 64, 3)
    kaiming_init(conv, a=0, mode='fan_in')

    # 检查权重不是全零或全一
    assert not torch.all(conv.weight == 0), "Weight should not be all zeros"
    assert not torch.all(conv.weight == 1), "Weight should not be all ones"

    # 检查 bias 是否为 0
    assert torch.all(conv.bias == 0), "Bias should be 0"

    print("✅ kaiming_init passed")


def test_xavier_init():
    """测试 Xavier 初始化"""
    print("Testing xavier_init...")

    linear = nn.Linear(128, 64)
    xavier_init(linear, gain=1.0, bias=0.1)

    assert not torch.all(linear.weight == 0), "Weight should not be all zeros"
    assert torch.all(linear.bias == 0.1), "Bias should be 0.1"

    print("✅ xavier_init passed")


def test_normal_init():
    """测试正态分布初始化"""
    print("Testing normal_init...")

    conv = nn.Conv2d(3, 64, 3)
    normal_init(conv, mean=0, std=0.01, bias=0)

    # 检查均值和标准差（粗略检查）
    weight_mean = conv.weight.mean().item()
    weight_std = conv.weight.std().item()

    assert abs(weight_mean) < 0.1, f"Mean should be close to 0, got {weight_mean}"
    assert abs(weight_std - 0.01) < 0.01, f"Std should be close to 0.01, got {weight_std}"

    print("✅ normal_init passed")


def test_default_init_weights():
    """测试默认初始化"""
    print("Testing default_init_weights...")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3)
            self.bn = nn.BatchNorm2d(64)
            self.linear = nn.Linear(64, 10)

    model = SimpleModel()
    default_init_weights(model, scale=0.1)

    # 检查 BatchNorm 的权重是否为 1
    assert torch.all(model.bn.weight == 1.0), "BatchNorm weight should be 1.0"
    assert torch.all(model.bn.bias == 0.0), "BatchNorm bias should be 0.0"

    print("✅ default_init_weights passed")


def test_checkpoint_save_load():
    """测试 checkpoint 保存和加载"""
    print("Testing checkpoint save and load...")

    # 创建模型
    model = nn.Linear(10, 5)
    original_weight = model.weight.clone()
    original_bias = model.bias.clone()

    # 保存
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"

        meta = {'epoch': 100, 'loss': 0.5}
        save_checkpoint(model, checkpoint_path, meta=meta)

        assert checkpoint_path.exists(), "Checkpoint file should exist"

        # 修改模型
        model.weight.data.fill_(0)
        model.bias.data.fill_(0)

        # 加载
        load_checkpoint(model, checkpoint_path, map_location='cpu')

        # 验证
        assert torch.allclose(model.weight, original_weight), "Weight should be restored"
        assert torch.allclose(model.bias, original_bias), "Bias should be restored"

        # 检查元信息
        checkpoint = torch.load(checkpoint_path)
        assert checkpoint['epoch'] == 100, "Epoch should be 100"
        assert checkpoint['loss'] == 0.5, "Loss should be 0.5"

    print("✅ checkpoint save/load passed")


def test_checkpoint_with_module_prefix():
    """测试加载带 'module.' 前缀的 checkpoint"""
    print("Testing checkpoint with 'module.' prefix...")

    model = nn.Linear(10, 5)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint_module.pth"

        # 创建带 'module.' 前缀的 state_dict
        state_dict = {f'module.{k}': v for k, v in model.state_dict().items()}
        torch.save({'state_dict': state_dict}, checkpoint_path)

        # 加载（应该自动移除 'module.' 前缀）
        load_checkpoint(model, checkpoint_path, map_location='cpu')

    print("✅ checkpoint with module prefix passed")


def test_logger():
    """测试日志功能"""
    print("Testing logger...")

    logger = get_logger('test_logger')
    logger.info("This is a test log message")

    print("✅ logger passed")


def test_get_module_device():
    """测试获取模块设备"""
    print("Testing get_module_device...")

    model = nn.Linear(10, 5)
    device = get_module_device(model)

    assert isinstance(device, torch.device), "Should return torch.device"
    assert device.type == 'cpu', "Should be on CPU"

    if torch.cuda.is_available():
        model_cuda = model.cuda()
        device_cuda = get_module_device(model_cuda)
        assert device_cuda.type == 'cuda', "Should be on CUDA"

    print("✅ get_module_device passed")


def test_set_requires_grad():
    """测试设置 requires_grad"""
    print("Testing set_requires_grad...")

    model = nn.Linear(10, 5)

    # 默认应该需要梯度
    assert all(p.requires_grad for p in model.parameters()), "Should require grad by default"

    # 冻结
    set_requires_grad(model, False)
    assert not any(p.requires_grad for p in model.parameters()), "Should not require grad"

    # 解冻
    set_requires_grad(model, True)
    assert all(p.requires_grad for p in model.parameters()), "Should require grad again"

    print("✅ set_requires_grad passed")


def test_is_batch_norm():
    """测试 BatchNorm 检查"""
    print("Testing is_batch_norm...")

    bn1d = nn.BatchNorm1d(64)
    bn2d = nn.BatchNorm2d(64)
    bn3d = nn.BatchNorm3d(64)
    conv = nn.Conv2d(3, 64, 3)

    assert is_batch_norm(bn1d), "Should be BatchNorm"
    assert is_batch_norm(bn2d), "Should be BatchNorm"
    assert is_batch_norm(bn3d), "Should be BatchNorm"
    assert not is_batch_norm(conv), "Should not be BatchNorm"

    print("✅ is_batch_norm passed")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Running pytorch_utils tests...")
    print("=" * 60)
    print()

    tests = [
        test_constant_init,
        test_kaiming_init,
        test_xavier_init,
        test_normal_init,
        test_default_init_weights,
        test_checkpoint_save_load,
        test_checkpoint_with_module_prefix,
        test_logger,
        test_get_module_device,
        test_set_requires_grad,
        test_is_batch_norm,
    ]

    failed = []

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed.append(test.__name__)
        print()

    print("=" * 60)
    if failed:
        print(f"❌ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"   - {name}")
        return False
    else:
        print(f"✅ All {len(tests)} tests passed!")
        return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
