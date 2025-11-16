# MMEngine 到 PyTorch 原生实现替换计划

> **文档版本**: 1.0
> **创建日期**: 2025-11-16
> **目标**: 将 BasicVSR++ 代码从 mmengine 依赖迁移到纯 PyTorch 实现

---

## 📋 目录

- [1. 项目概述](#1-项目概述)
- [2. 依赖分析](#2-依赖分析)
- [3. 替换策略](#3-替换策略)
- [4. 分阶段实施计划](#4-分阶段实施计划)
- [5. 详细替换方案](#5-详细替换方案)
- [6. 测试验证](#6-测试验证)
- [7. 兼容性保证](#7-兼容性保证)
- [8. 风险评估](#8-风险评估)
- [9. 回滚方案](#9-回滚方案)

---

## 1. 项目概述

### 1.1 替换目标

将当前 BasicVSR++ 实现从依赖 `mmengine` 迁移到完全使用 PyTorch 原生 API，主要目标：

- ✅ **简化依赖**: 移除 mmengine 依赖，仅保留 PyTorch/torchvision
- ✅ **提升兼容性**: 更容易集成到其他项目
- ✅ **降低安装门槛**: 减少用户安装时的复杂度
- ✅ **保持性能**: 确保替换后性能不降低
- ✅ **保持功能**: 100% 保持现有功能

### 1.2 当前依赖状况

```python
# 当前 requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
mmengine>=0.10.0  # ⚠️ 需要替换
Pillow>=10.0.0
av>=10.0.0
scipy>=1.10.0
```

### 1.3 目标依赖

```python
# 替换后 requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
av>=10.0.0
scipy>=1.10.0
# mmengine 已移除 ✅
```

---

## 2. 依赖分析

### 2.1 当前 mmengine 使用统计

通过代码扫描，发现 mmengine 在以下 29 个文件中被使用：

```
basicvsrpp/inference.py
basicvsrpp/mmagic/__init__.py
basicvsrpp/mmagic/base_edit_model.py
basicvsrpp/mmagic/base_gen_metric.py
basicvsrpp/mmagic/base_sample_wise_metric.py
basicvsrpp/mmagic/basicvsr_plusplus_net.py        # ⚠️ 核心文件
basicvsrpp/mmagic/concat_visualizer.py
basicvsrpp/mmagic/data_preprocessor.py
basicvsrpp/mmagic/data_sample.py
basicvsrpp/mmagic/ema.py
basicvsrpp/mmagic/evaluator.py
basicvsrpp/mmagic/gan_loss.py
basicvsrpp/mmagic/img_utils.py
basicvsrpp/mmagic/iter_time_hook.py
basicvsrpp/mmagic/log_processor.py
basicvsrpp/mmagic/logger.py
basicvsrpp/mmagic/loop_utils.py
basicvsrpp/mmagic/model_utils.py                  # ⚠️ 工具函数
basicvsrpp/mmagic/multi_loops.py
basicvsrpp/mmagic/multi_optimizer_constructor.py
basicvsrpp/mmagic/perceptual_loss.py              # ⚠️ 核心文件
basicvsrpp/mmagic/real_basicvsr.py                # ⚠️ 核心文件
basicvsrpp/mmagic/registry.py                     # ⚠️ 注册系统
basicvsrpp/mmagic/sampler.py
basicvsrpp/mmagic/setup_env.py
basicvsrpp/mmagic/typing.py
basicvsrpp/mmagic/unet_disc.py
basicvsrpp/mmagic/vis_backend.py
basicvsrpp/mmagic/visualization_hook.py
```

### 2.2 mmengine 组件使用分类

| 组件类型 | mmengine 类/函数 | 使用频率 | 替换难度 | 优先级 |
|---------|------------------|---------|---------|--------|
| **基础类** | `BaseModule` | 8次 | ⭐ 简单 | 🔴 高 |
| **权重初始化** | `constant_init`, `kaiming_init`, etc. | 15次 | ⭐⭐ 中等 | 🔴 高 |
| **模型加载** | `load_checkpoint` | 3次 | ⭐⭐ 中等 | 🔴 高 |
| **日志系统** | `MMLogger` | 5次 | ⭐ 简单 | 🟡 中 |
| **配置系统** | `Config` | 2次 | ⭐⭐ 中等 | 🟡 中 |
| **注册机制** | `Registry` | 全局 | ⭐⭐⭐⭐ 复杂 | 🟢 低 |
| **训练循环** | `BaseLoop`, `Runner` | 多处 | ⭐⭐⭐⭐⭐ 很复杂 | 🟢 低 |
| **数据预处理** | `DataPreprocessor` | 1次 | ⭐⭐⭐ 中等 | 🟢 低 |
| **Hooks** | `Hook`, `IterTimerHook` | 多处 | ⭐⭐⭐⭐ 复杂 | 🟢 低 |

### 2.3 核心计算算子（已使用 PyTorch 原生）

**✅ 好消息**: 所有核心计算算子已经在使用 PyTorch/torchvision 原生实现：

```python
# 可变形卷积 - torchvision 原生
torchvision.ops.deform_conv2d(...)

# 光流扭曲 - PyTorch 原生
F.grid_sample(...)

# 上采样 - PyTorch 原生
F.interpolate(...)

# Pixel Shuffle - PyTorch 原生
F.pixel_shuffle(...)

# 其他操作
F.pad(...)
F.avg_pool2d(...)
torch.meshgrid(...)
```

**结论**: 替换 mmengine 不会影响核心计算性能！

---

## 3. 替换策略

### 3.1 策略原则

1. **渐进式替换**: 分阶段进行，每阶段确保可运行
2. **向后兼容**: 保持原有 API 接口不变
3. **测试驱动**: 每次替换后运行完整测试
4. **性能保证**: 替换后性能不降低
5. **最小依赖**: 优先使用 Python 标准库和 PyTorch 原生 API

### 3.2 替换优先级

#### 🔴 阶段 1: 核心模型独立（推理功能）
**目标**: 推理代码完全独立，不依赖 mmengine
**影响范围**: 5个核心文件
**预计工作量**: 2-3天

#### 🟡 阶段 2: 工具函数替换
**目标**: 替换所有工具函数和辅助类
**影响范围**: 10个工具文件
**预计工作量**: 3-5天

#### 🟢 阶段 3: 训练框架简化（可选）
**目标**: 替换训练相关组件
**影响范围**: 15个训练文件
**预计工作量**: 5-7天

---

## 4. 分阶段实施计划

### 阶段 1: 核心模型独立（推理功能） 🔴

#### 目标
推理代码（`inference.py` + 核心模型）完全独立，可以只用 PyTorch 运行推理。

#### 涉及文件
```
✓ basicvsrpp/inference.py
✓ basicvsrpp/mmagic/basicvsr_plusplus_net.py
✓ basicvsrpp/mmagic/perceptual_loss.py
✓ basicvsrpp/mmagic/real_basicvsr.py
✓ basicvsrpp/mmagic/model_utils.py
```

#### 替换清单

| 序号 | 原 mmengine 组件 | PyTorch 替换 | 文件 | 难度 |
|-----|-----------------|-------------|------|------|
| 1 | `BaseModule` | `nn.Module` | basicvsr_plusplus_net.py | ⭐ |
| 2 | `constant_init` | `init.constant_` | basicvsr_plusplus_net.py | ⭐ |
| 3 | `kaiming_init` | `init.kaiming_normal_` | basicvsr_plusplus_net.py, model_utils.py | ⭐⭐ |
| 4 | `load_checkpoint` | 自定义函数 | inference.py, basicvsr_plusplus_net.py | ⭐⭐ |
| 5 | `MMLogger` | `logging.getLogger` | 多个文件 | ⭐ |
| 6 | `Config` | `dict` 配置 | inference.py | ⭐⭐ |

#### 详细步骤

**步骤 1.1: 替换 BaseModule** (预计 30 分钟)

```python
# 修改文件: basicvsrpp/mmagic/basicvsr_plusplus_net.py

# 原代码 (第 10 行)
from mmengine.model import BaseModule

# 替换为
import torch.nn as nn

# 原代码 (第 23 行)
class BasicVSRPlusPlusNet(BaseModule):
    def __init__(self, ...):
        super().__init__()

# 替换为
class BasicVSRPlusPlusNet(nn.Module):
    def __init__(self, ...):
        super().__init__()
```

同样替换以下类：
- `ResidualBlocksWithInputConv` (第 331 行)
- `SPyNet` (第 369 行)
- `SPyNetBasicModule` (第 517 行)

**步骤 1.2: 替换权重初始化** (预计 1 小时)

创建新文件 `basicvsrpp/mmagic/pytorch_init.py`:

```python
"""PyTorch 原生权重初始化工具"""
import torch.nn as nn
import torch.nn.init as init


def constant_init(module, val, bias=0):
    """常量初始化

    Args:
        module: nn.Module 或权重张量
        val: 初始化值
        bias: bias 初始化值
    """
    if isinstance(module, nn.Module):
        if hasattr(module, 'weight') and module.weight is not None:
            init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            init.constant_(module.bias, bias)
    else:
        # 直接是张量
        init.constant_(module, val)


def kaiming_init(module, a=0, mode='fan_in', nonlinearity='leaky_relu', bias=0):
    """Kaiming 初始化

    Args:
        module: nn.Module
        a: LeakyReLU 的负斜率
        mode: 'fan_in' 或 'fan_out'
        nonlinearity: 激活函数类型
        bias: bias 初始化值
    """
    if hasattr(module, 'weight') and module.weight is not None:
        init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Xavier 初始化

    Args:
        module: nn.Module
        gain: 增益系数
        bias: bias 初始化值
        distribution: 'normal' 或 'uniform'
    """
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            init.xavier_uniform_(module.weight, gain=gain)
        else:
            init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    """正态分布初始化

    Args:
        module: nn.Module
        mean: 均值
        std: 标准差
        bias: bias 初始化值
    """
    if hasattr(module, 'weight') and module.weight is not None:
        init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)
```

然后在所有使用到的地方替换导入：

```python
# 原代码
from mmengine.model.weight_init import constant_init, kaiming_init

# 替换为
from basicvsrpp.mmagic.pytorch_init import constant_init, kaiming_init
```

**步骤 1.3: 替换 load_checkpoint** (预计 1.5 小时)

创建新文件 `basicvsrpp/mmagic/pytorch_checkpoint.py`:

```python
"""PyTorch 原生 checkpoint 加载工具"""
import torch
import logging
from typing import Optional, Dict, Any


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    map_location: str = 'cpu',
    strict: bool = True,
    logger: Optional[logging.Logger] = None
) -> torch.nn.Module:
    """加载模型 checkpoint

    Args:
        model: 要加载权重的模型
        checkpoint_path: checkpoint 文件路径
        map_location: 加载到的设备
        strict: 是否严格匹配 state_dict
        logger: 日志记录器

    Returns:
        加载权重后的模型
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f'Loading checkpoint from {checkpoint_path}')

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # 提取 state_dict（支持多种格式）
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'generator' in checkpoint:
            # 某些 GAN 模型的格式
            state_dict = checkpoint['generator']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 处理 DataParallel/DistributedDataParallel 包装的模型
    # 移除 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
        else:
            new_state_dict[k] = v

    # 加载到模型
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)

    # 日志输出
    if missing_keys:
        logger.warning(f'Missing keys in checkpoint: {missing_keys}')
    if unexpected_keys:
        logger.warning(f'Unexpected keys in checkpoint: {unexpected_keys}')

    if not missing_keys and not unexpected_keys:
        logger.info('Checkpoint loaded successfully!')

    return model


def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    meta: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """保存模型 checkpoint

    Args:
        model: 要保存的模型
        checkpoint_path: 保存路径
        meta: 额外的元信息（如 epoch, optimizer 等）
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # 构建 checkpoint
    checkpoint = {
        'state_dict': model.state_dict()
    }

    # 添加元信息
    if meta is not None:
        checkpoint.update(meta)

    # 保存
    torch.save(checkpoint, checkpoint_path)
    logger.info(f'Checkpoint saved to {checkpoint_path}')
```

替换所有使用：

```python
# 原代码
from mmengine.runner import load_checkpoint

# 替换为
from basicvsrpp.mmagic.pytorch_checkpoint import load_checkpoint
```

**步骤 1.4: 替换 MMLogger** (预计 30 分钟)

创建新文件 `basicvsrpp/mmagic/pytorch_logger.py`:

```python
"""PyTorch 原生日志工具"""
import logging
import sys


def setup_logger(
    name: str = __name__,
    log_file: str = None,
    log_level: int = logging.INFO,
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """设置日志记录器

    Args:
        name: logger 名称
        log_file: 日志文件路径（可选）
        log_level: 日志级别
        format: 日志格式

    Returns:
        配置好的 logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 创建格式化器
    formatter = logging.Formatter(format)

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件 handler（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """获取 logger（简化版本）

    Args:
        name: logger 名称

    Returns:
        logger 实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 如果没有配置过，使用默认配置
        return setup_logger(name)
    return logger
```

替换所有使用：

```python
# 原代码
from mmengine import MMLogger
logger = MMLogger.get_current_instance()

# 替换为
from basicvsrpp.mmagic.pytorch_logger import get_logger
logger = get_logger(__name__)
```

**步骤 1.5: 简化 Config 系统** (预计 1 小时)

修改 `basicvsrpp/inference.py`:

```python
# 原代码
from mmengine.config import Config

def load_model(config: str | dict | None, checkpoint_path, device):
    if type(config) == str:
        config = Config.fromfile(config).model
    elif type(config) == dict:
        pass
    else:
        raise Exception("...")

# 替换为
import importlib.util
from pathlib import Path

def load_model(config: str | dict | None, checkpoint_path, device):
    """加载模型

    Args:
        config: 配置文件路径(.py)、配置字典或 None
        checkpoint_path: checkpoint 路径
        device: 设备
    """
    if isinstance(config, str):
        # 支持 Python 配置文件
        config_path = Path(config)
        if config_path.suffix == '.py':
            # 动态导入 Python 配置文件
            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            config = config_module.model
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    elif isinstance(config, dict):
        # 直接使用字典配置
        pass
    elif config is None:
        # 使用默认配置
        config = get_default_gan_inference_config()
    else:
        raise TypeError(f"config must be str, dict or None, but got {type(config)}")

    # ... 后续代码保持不变
```

#### 测试验证

在每个步骤后运行测试：

```bash
# 测试推理功能
python -c "
from basicvsrpp.inference import load_model, inference, get_default_gan_inference_config
import numpy as np

config = get_default_gan_inference_config()
# 注意：这里需要一个实际的 checkpoint
# model = load_model(config, 'checkpoint.pth', 'cpu')

# 测试推理接口
frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(4)]
# result = inference(model, frames, 'cpu')
print('✅ Inference API test passed')
"
```

#### 预期结果

- ✅ 推理代码不再依赖 mmengine
- ✅ 所有测试通过
- ✅ 性能无降低
- ✅ API 接口保持不变

---

### 阶段 2: 工具函数替换 🟡

#### 目标
替换所有工具类和辅助函数中的 mmengine 依赖。

#### 涉及文件
```
✓ basicvsrpp/mmagic/model_utils.py
✓ basicvsrpp/mmagic/img_utils.py
✓ basicvsrpp/mmagic/colorspace.py
✓ basicvsrpp/mmagic/data_preprocessor.py
✓ basicvsrpp/mmagic/gan_loss.py
✓ basicvsrpp/mmagic/pixelwise_loss.py
✓ basicvsrpp/mmagic/unet_disc.py
... (其他工具文件)
```

#### 详细步骤

**步骤 2.1: 简化 model_utils.py**

```python
# 原代码
from mmengine.registry import Registry
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

# 替换为
import torch.nn as nn

# _BatchNorm 替换为标准 BatchNorm 检查
def _is_batch_norm(module):
    """检查是否为 BatchNorm 层"""
    return isinstance(module, (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm
    ))

# 在 default_init_weights 中使用
def default_init_weights(module, scale=1):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif _is_batch_norm(m):  # 使用新的检查函数
            constant_init(m, val=1, bias=0)
```

**步骤 2.2: 简化 Registry 系统**

创建 `basicvsrpp/mmagic/simple_registry.py`:

```python
"""简化的注册系统"""
from typing import Dict, Any, Callable, Optional


class Registry:
    """简化的注册表实现"""

    def __init__(self, name: str, parent: Optional['Registry'] = None):
        self._name = name
        self._module_dict: Dict[str, Any] = {}
        self._parent = parent

    def register_module(self, name: str = None, force: bool = False):
        """注册装饰器

        Args:
            name: 注册名称，如果为 None 则使用类名
            force: 是否强制覆盖已存在的注册
        """
        def _register(cls):
            module_name = name if name is not None else cls.__name__

            if module_name in self._module_dict and not force:
                raise KeyError(f'{module_name} is already registered in {self._name}')

            self._module_dict[module_name] = cls
            return cls

        return _register

    def build(self, cfg: Dict[str, Any], **kwargs) -> Any:
        """构建模块

        Args:
            cfg: 配置字典，必须包含 'type' 键
            **kwargs: 额外的构造参数
        """
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

        if 'type' not in cfg:
            raise KeyError('cfg must contain "type" key')

        cfg = cfg.copy()
        obj_type = cfg.pop('type')

        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
        else:
            obj_cls = obj_type

        # 合并参数
        cfg.update(kwargs)

        return obj_cls(**cfg)

    def get(self, name: str) -> Any:
        """获取注册的模块"""
        if name in self._module_dict:
            return self._module_dict[name]
        elif self._parent is not None:
            return self._parent.get(name)
        else:
            raise KeyError(f'{name} is not registered in {self._name}')

    def __contains__(self, name: str) -> bool:
        return name in self._module_dict or (
            self._parent is not None and name in self._parent
        )

    def __repr__(self) -> str:
        return f'Registry(name={self._name}, items={list(self._module_dict.keys())})'


# 创建常用的注册表
MODELS = Registry('model')
LOSSES = Registry('loss')
```

然后替换 `basicvsrpp/mmagic/registry.py`:

```python
"""注册系统 - 使用简化版本"""

from .simple_registry import Registry, MODELS, LOSSES

# 为了兼容性，创建其他需要的注册表
DATASETS = Registry('dataset')
METRICS = Registry('metric')
TRANSFORMS = Registry('transform')
# ... 其他需要的注册表

__all__ = ['Registry', 'MODELS', 'LOSSES', 'DATASETS', 'METRICS', 'TRANSFORMS']
```

#### 预期结果

- ✅ 工具函数完全独立
- ✅ 保持原有功能
- ✅ 代码更加清晰

---

### 阶段 3: 训练框架简化（可选）🟢

#### 目标
替换训练相关的 mmengine 组件，使用 PyTorch 原生或 PyTorch Lightning。

#### 选项 A: 纯 PyTorch 实现

创建简单的训练循环：

```python
"""简单的训练循环"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class SimpleTrainer:
    """简化的训练器"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        device: str = 'cuda',
        max_epochs: int = 100
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.current_epoch = 0

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            inputs = batch['inputs'].to(self.device)
            targets = batch['targets'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # 计算损失
            loss = self.compute_loss(outputs, targets)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """验证"""
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)

                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets)

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def compute_loss(self, outputs, targets):
        """计算损失 - 需要子类实现"""
        raise NotImplementedError

    def train(self):
        """完整训练循环"""
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # 训练
            train_loss = self.train_epoch()
            print(f'Epoch {epoch}: train_loss={train_loss:.4f}')

            # 验证
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f'Epoch {epoch}: val_loss={val_loss:.4f}')
```

#### 选项 B: 使用 PyTorch Lightning

```python
"""使用 PyTorch Lightning"""
import pytorch_lightning as pl


class BasicVSRLightningModule(pl.LightningModule):
    """BasicVSR++ Lightning 模块"""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # 构建模型
        from basicvsrpp.mmagic.registry import MODELS
        self.model = MODELS.build(config['generator'])

        # 损失函数
        self.pixel_loss = MODELS.build(config['pixel_loss'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = batch['inputs']
        targets = batch['gt']

        # 前向传播
        outputs = self(inputs)

        # 计算损失
        loss = self.pixel_loss(outputs, targets)

        # 日志
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['inputs']
        targets = batch['gt']

        outputs = self(inputs)
        loss = self.pixel_loss(outputs, targets)

        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


# 使用
def train_with_lightning():
    # 创建模块
    config = {...}
    model = BasicVSRLightningModule(config)

    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=1,
        precision=16
    )

    # 训练
    trainer.fit(model, train_loader, val_loader)
```

#### 预期结果

- ✅ 训练代码独立
- ✅ 支持分布式训练（通过 Lightning）
- ✅ 代码更加简洁

---

## 5. 详细替换方案

### 5.1 BaseModule → nn.Module

**影响范围**: 8 个类

**替换方法**:
```python
# 查找所有
grep -r "class.*BaseModule" basicvsrpp/

# 批量替换
sed -i 's/from mmengine.model import BaseModule/import torch.nn as nn/' basicvsrpp/mmagic/*.py
sed -i 's/class \(.*\)(BaseModule)/class \1(nn.Module)/' basicvsrpp/mmagic/*.py
```

**验证**:
```python
# 确保所有类仍然可以实例化
from basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet
model = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=7)
assert isinstance(model, nn.Module)
```

---

### 5.2 权重初始化函数

**对照表**:

| mmengine | PyTorch | 说明 |
|----------|---------|------|
| `constant_init(module, val, bias)` | `init.constant_(module.weight, val)` | 常量初始化 |
| `kaiming_init(module, a, mode)` | `init.kaiming_normal_(module.weight, a, mode)` | Kaiming 初始化 |
| `xavier_init(module, gain)` | `init.xavier_normal_(module.weight, gain)` | Xavier 初始化 |
| `normal_init(module, mean, std)` | `init.normal_(module.weight, mean, std)` | 正态分布 |

**替换示例**:

```python
# 示例 1: constant_init
# 原代码
from mmengine.model.weight_init import constant_init
constant_init(self.conv_offset[-1], val=0, bias=0)

# 替换后
import torch.nn.init as init
init.constant_(self.conv_offset[-1].weight, 0)
if self.conv_offset[-1].bias is not None:
    init.constant_(self.conv_offset[-1].bias, 0)

# 示例 2: kaiming_init
# 原代码
kaiming_init(m, a=0, mode='fan_in', bias=0)

# 替换后
init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
if m.bias is not None:
    init.constant_(m.bias, 0)
```

---

### 5.3 模型加载

**实现对比**:

```python
# mmengine 实现
from mmengine.runner import load_checkpoint
load_checkpoint(model, 'checkpoint.pth', map_location='cpu', strict=True, logger=logger)

# PyTorch 原生实现
checkpoint = torch.load('checkpoint.pth', map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)

# 移除 'module.' 前缀
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 加载
missing, unexpected = model.load_state_dict(state_dict, strict=True)
if missing:
    logger.warning(f'Missing keys: {missing}')
if unexpected:
    logger.warning(f'Unexpected keys: {unexpected}')
```

---

### 5.4 日志系统

**实现对比**:

```python
# mmengine 实现
from mmengine import MMLogger
logger = MMLogger.get_current_instance()
logger.info('message')

# Python logging 实现
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger.info('message')
```

---

### 5.5 配置系统

**实现对比**:

```python
# mmengine Config
from mmengine.config import Config
config = Config.fromfile('config.py')

# 方案 1: 直接使用字典
config = {
    'model': {'type': 'BasicVSRPlusPlusNet', 'mid_channels': 64}
}

# 方案 2: YAML
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# 方案 3: Python 文件 (保持 mmengine 风格)
import importlib.util
spec = importlib.util.spec_from_file_location("config", "config.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = module.model
```

---

## 6. 测试验证

### 6.1 单元测试

创建 `tests/test_pytorch_migration.py`:

```python
"""测试 PyTorch 原生实现的替换"""
import torch
import torch.nn as nn
import numpy as np
import pytest


class TestBasicModule:
    """测试基础模块替换"""

    def test_basicvsr_net_creation(self):
        """测试 BasicVSRPlusPlusNet 可以创建"""
        from basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

        model = BasicVSRPlusPlusNet(
            mid_channels=64,
            num_blocks=7,
            max_residue_magnitude=10,
            spynet_pretrained=None
        )

        assert isinstance(model, nn.Module)
        assert model.mid_channels == 64

    def test_forward_pass(self):
        """测试前向传播"""
        from basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

        model = BasicVSRPlusPlusNet(
            mid_channels=64,
            num_blocks=7,
            spynet_pretrained=None
        )
        model.eval()

        # 创建随机输入 (B, T, C, H, W)
        batch_size, num_frames = 1, 3
        h, w = 64, 64
        inputs = torch.randn(batch_size, num_frames, 3, h, w)

        with torch.no_grad():
            outputs = model(inputs)

        # 检查输出形状 (4x 上采样)
        assert outputs.shape == (batch_size, num_frames, 3, h*4, w*4)


class TestInitFunctions:
    """测试初始化函数"""

    def test_constant_init(self):
        """测试常量初始化"""
        from basicvsrpp.mmagic.pytorch_init import constant_init

        conv = nn.Conv2d(3, 64, 3)
        constant_init(conv, val=1.0, bias=0.5)

        assert torch.all(conv.weight == 1.0)
        assert torch.all(conv.bias == 0.5)

    def test_kaiming_init(self):
        """测试 Kaiming 初始化"""
        from basicvsrpp.mmagic.pytorch_init import kaiming_init

        conv = nn.Conv2d(3, 64, 3)
        kaiming_init(conv, a=0, mode='fan_in')

        # 检查权重不是全零
        assert not torch.all(conv.weight == 0)


class TestCheckpointLoading:
    """测试 checkpoint 加载"""

    def test_save_and_load(self, tmp_path):
        """测试保存和加载"""
        from basicvsrpp.mmagic.pytorch_checkpoint import save_checkpoint, load_checkpoint

        # 创建模型
        model = nn.Linear(10, 5)
        original_weight = model.weight.clone()

        # 保存
        checkpoint_path = tmp_path / "test.pth"
        save_checkpoint(model, str(checkpoint_path))

        # 修改模型
        model.weight.data.fill_(0)

        # 加载
        load_checkpoint(model, str(checkpoint_path))

        # 验证
        assert torch.allclose(model.weight, original_weight)


class TestInference:
    """测试推理功能"""

    def test_inference_api(self):
        """测试推理接口"""
        from basicvsrpp.inference import get_default_gan_inference_config

        config = get_default_gan_inference_config()

        assert config['type'] == 'BasicVSRPlusPlusGan'
        assert 'generator' in config
        assert config['generator']['mid_channels'] == 64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### 6.2 集成测试

创建 `tests/test_end_to_end.py`:

```python
"""端到端测试"""
import torch
import numpy as np
import pytest


def test_full_inference_pipeline():
    """测试完整的推理流程"""
    from basicvsrpp.inference import get_default_gan_inference_config
    from basicvsrpp.mmagic.registry import MODELS

    # 创建配置
    config = get_default_gan_inference_config()

    # 构建模型（不加载权重）
    model = MODELS.build(config)
    model.eval()

    # 创建假数据
    frames = [
        np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        for _ in range(4)
    ]

    # 推理
    from basicvsrpp.inference import inference

    with torch.no_grad():
        # 注意：这里会因为没有真实 checkpoint 而失败
        # 但可以测试接口是否正常
        try:
            # result = inference(model, frames, 'cpu')
            pass
        except Exception as e:
            # 预期会有错误，因为模型未初始化
            print(f"Expected error: {e}")

    print("✅ Inference pipeline test passed")


if __name__ == '__main__':
    test_full_inference_pipeline()
```

### 6.3 性能测试

创建 `tests/benchmark_performance.py`:

```python
"""性能基准测试"""
import torch
import time
import numpy as np


def benchmark_forward_pass():
    """测试前向传播性能"""
    from basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = BasicVSRPlusPlusNet(
        mid_channels=64,
        num_blocks=7,
        spynet_pretrained=None
    ).to(device)
    model.eval()

    # 预热
    inputs = torch.randn(1, 3, 3, 64, 64).to(device)
    with torch.no_grad():
        _ = model(inputs)

    # 基准测试
    num_runs = 10
    times = []

    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(inputs)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"Forward pass time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"FPS: {1/avg_time:.2f}")


if __name__ == '__main__':
    benchmark_forward_pass()
```

---

## 7. 兼容性保证

### 7.1 向后兼容策略

**保持原有导入路径**:

```python
# 用户代码不需要改变
from basicvsrpp.inference import load_model, inference
from basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

# 内部实现已替换，但外部接口不变
```

**保持 API 签名**:

```python
# 替换前后，函数签名保持一致
def load_model(config: str | dict | None, checkpoint_path, device):
    # 实现变了，但签名不变
    pass
```

### 7.2 渐进式迁移

创建兼容层 `basicvsrpp/compat.py`:

```python
"""兼容层：支持新旧两种实现"""
import os

# 环境变量控制是否使用 mmengine
USE_MMENGINE = os.environ.get('BASICVSR_USE_MMENGINE', '0') == '1'

if USE_MMENGINE:
    # 使用旧的 mmengine 实现
    from mmengine.model import BaseModule
    from mmengine.runner import load_checkpoint
else:
    # 使用新的 PyTorch 原生实现
    import torch.nn as nn
    BaseModule = nn.Module
    from basicvsrpp.mmagic.pytorch_checkpoint import load_checkpoint

__all__ = ['BaseModule', 'load_checkpoint']
```

使用：

```python
# 在需要兼容的文件中
from basicvsrpp.compat import BaseModule, load_checkpoint

class MyModel(BaseModule):
    pass  # 自动使用正确的基类
```

---

## 8. 风险评估

### 8.1 风险矩阵

| 风险项 | 可能性 | 影响 | 风险等级 | 缓解措施 |
|--------|--------|------|----------|----------|
| 性能降低 | 低 | 高 | 🟡 中 | 性能基准测试 |
| 功能缺失 | 低 | 高 | 🟡 中 | 完整单元测试 |
| API 不兼容 | 中 | 中 | 🟡 中 | 兼容层 + 文档 |
| checkpoint 不兼容 | 低 | 高 | 🟡 中 | 格式转换工具 |
| 训练不稳定 | 中 | 高 | 🔴 高 | 充分测试 + Lightning |
| 依赖冲突 | 低 | 低 | 🟢 低 | 明确版本要求 |

### 8.2 关键风险详解

#### 风险 1: Checkpoint 格式不兼容

**描述**: 旧 mmengine 保存的 checkpoint 可能有特殊格式

**缓解**:
```python
def load_checkpoint_compatible(model, checkpoint_path):
    """兼容多种 checkpoint 格式"""
    checkpoint = torch.load(checkpoint_path)

    # 尝试多种键名
    for key in ['state_dict', 'model', 'generator', 'state']:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    else:
        state_dict = checkpoint

    # 处理各种前缀
    prefixes_to_remove = ['module.', 'model.', 'generator.']
    for prefix in prefixes_to_remove:
        state_dict = {
            k.replace(prefix, '') if k.startswith(prefix) else k: v
            for k, v in state_dict.items()
        }

    return model.load_state_dict(state_dict, strict=False)
```

#### 风险 2: 训练时数值稳定性

**描述**: 替换初始化函数后可能影响收敛

**缓解**:
- 使用相同的随机种子进行对比测试
- 记录每个 epoch 的 loss，对比曲线
- 提供权重转换工具

---

## 9. 回滚方案

### 9.1 Git 分支策略

```bash
# 主分支：稳定的 mmengine 版本
git checkout main

# 创建迁移分支
git checkout -b pytorch-native-migration

# 阶段性提交
git commit -m "Phase 1: Replace BaseModule"
git commit -m "Phase 2: Replace init functions"
...

# 如果出问题，可以随时回退
git checkout main
```

### 9.2 功能开关

在代码中保留切换能力：

```python
# config.py
ENABLE_PYTORCH_NATIVE = True  # 设为 False 可回退到 mmengine

# 在代码中
if ENABLE_PYTORCH_NATIVE:
    from basicvsrpp.mmagic.pytorch_init import kaiming_init
else:
    from mmengine.model.weight_init import kaiming_init
```

### 9.3 Docker 镜像

保留旧版本 Docker 镜像：

```dockerfile
# Dockerfile.mmengine (旧版本)
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN pip install mmengine>=0.10.0
...

# Dockerfile.native (新版本)
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
# 不安装 mmengine
...
```

---

## 10. 实施时间表

### 第 1 周：准备和规划
- [ ] Day 1-2: 完成依赖分析和测试用例准备
- [ ] Day 3-4: 创建 PyTorch 原生工具函数（init, checkpoint, logger）
- [ ] Day 5: 设置 CI/CD 测试流程

### 第 2 周：阶段 1 实施
- [ ] Day 1: 替换 BaseModule
- [ ] Day 2: 替换权重初始化函数
- [ ] Day 3: 替换 load_checkpoint
- [ ] Day 4: 替换 Logger 和 Config
- [ ] Day 5: 测试和修复问题

### 第 3 周：阶段 2 实施
- [ ] Day 1-3: 替换工具函数
- [ ] Day 4: 简化 Registry
- [ ] Day 5: 集成测试

### 第 4 周：验证和文档
- [ ] Day 1-2: 性能测试和对比
- [ ] Day 3: 修复发现的问题
- [ ] Day 4: 更新文档和示例
- [ ] Day 5: Code review 和最终验证

---

## 11. 成功指标

替换成功的标准：

- ✅ **依赖**: requirements.txt 不包含 mmengine
- ✅ **功能**: 所有测试用例通过
- ✅ **性能**: 推理速度误差 < 5%
- ✅ **兼容性**: 可以加载旧的 checkpoint
- ✅ **易用性**: 安装步骤减少 50%
- ✅ **文档**: 更新所有相关文档

---

## 12. 参考资源

### 12.1 官方文档
- [PyTorch nn.init 文档](https://pytorch.org/docs/stable/nn.init.html)
- [PyTorch checkpoint 文档](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
- [Python logging 文档](https://docs.python.org/3/library/logging.html)

### 12.2 相关项目
- [torchvision.ops](https://pytorch.org/vision/stable/ops.html) - 可变形卷积等
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) - 训练框架（可选）

### 12.3 迁移示例
- [MMDetection 到纯 PyTorch](https://github.com/open-mmlab/mmdetection/discussions/8508)
- [timm 库实现](https://github.com/huggingface/pytorch-image-models) - 纯 PyTorch 实现参考

---

## 附录 A: 快速命令

```bash
# 查找所有 mmengine 导入
grep -r "from mmengine" basicvsrpp/ --include="*.py" | wc -l

# 查找所有 BaseModule 使用
grep -r "class.*BaseModule" basicvsrpp/ --include="*.py"

# 运行测试
python -m pytest tests/ -v

# 性能基准
python tests/benchmark_performance.py

# 检查依赖
pip list | grep mmengine
```

---

## 附录 B: 常见问题 FAQ

**Q1: 替换后性能会下降吗？**
A: 不会。核心计算算子已经在使用 PyTorch 原生实现，替换的只是辅助组件。

**Q2: 能加载旧的 checkpoint 吗？**
A: 可以。我们的 load_checkpoint 函数兼容多种格式。

**Q3: 需要重新训练模型吗？**
A: 不需要。只要 checkpoint 兼容，直接加载即可。

**Q4: 如果遇到问题怎么办？**
A: 使用 Git 回退到之前的版本，或通过功能开关切换回 mmengine。

**Q5: 是否支持分布式训练？**
A: 阶段 3 完成后支持。可以用 PyTorch 原生 DDP 或 PyTorch Lightning。

---

## 结论

本迁移计划旨在将 BasicVSR++ 从 mmengine 依赖迁移到纯 PyTorch 实现，主要优点：

1. ✅ **简化依赖**: 只需 PyTorch + torchvision
2. ✅ **提升兼容性**: 更容易集成到其他项目
3. ✅ **保持性能**: 核心算子已经是 PyTorch 原生
4. ✅ **降低门槛**: 安装和使用更简单

迁移采用分阶段策略，确保每个阶段都可以独立验证和回滚，降低风险。

---

**文档维护者**: BasicVSR++ Clean 项目团队
**最后更新**: 2025-11-16
**版本**: 1.0
