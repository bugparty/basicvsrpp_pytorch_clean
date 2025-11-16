# BasicVSR++ 迁移快速指南

> 📖 完整文档请参考: [MMENGINE_TO_PYTORCH_MIGRATION.md](MMENGINE_TO_PYTORCH_MIGRATION.md)

## 🎯 一分钟了解

将 mmengine 依赖替换为 PyTorch 原生实现，简化依赖并提升兼容性。

## 📊 替换对照表

### 常用组件替换

| mmengine | PyTorch 原生 | 说明 |
|----------|-------------|------|
| `BaseModule` | `nn.Module` | 基础模块类 |
| `constant_init(m, val)` | `init.constant_(m.weight, val)` | 常量初始化 |
| `kaiming_init(m)` | `init.kaiming_normal_(m.weight)` | Kaiming 初始化 |
| `load_checkpoint(...)` | `torch.load(...)` + `load_state_dict(...)` | 加载权重 |
| `MMLogger` | `logging.getLogger(...)` | 日志系统 |
| `Config.fromfile(...)` | `dict` 或 `yaml.load(...)` | 配置系统 |

## 🚀 快速开始

### 步骤 1: 创建辅助工具

创建 `basicvsrpp/mmagic/pytorch_utils.py`:

```python
"""PyTorch 原生工具集合"""
import torch
import torch.nn as nn
import torch.nn.init as init
import logging
from typing import Optional, Dict, Any


# ============ 权重初始化 ============
def constant_init(module, val, bias=0):
    """常量初始化"""
    if hasattr(module, 'weight') and module.weight is not None:
        init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_in', nonlinearity='leaky_relu', bias=0):
    """Kaiming 初始化"""
    if hasattr(module, 'weight') and module.weight is not None:
        init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


# ============ Checkpoint 加载 ============
def load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True, logger=None):
    """加载 checkpoint"""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f'Loading checkpoint from {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # 提取 state_dict
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict',
                     checkpoint.get('model',
                     checkpoint.get('generator', checkpoint)))
    else:
        state_dict = checkpoint

    # 移除 'module.' 前缀
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 加载
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        logger.warning(f'Missing keys: {missing}')
    if unexpected:
        logger.warning(f'Unexpected keys: {unexpected}')

    return model


# ============ 日志工具 ============
def get_logger(name=__name__, level=logging.INFO):
    """获取 logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
```

### 步骤 2: 批量替换导入

在需要修改的文件中：

```python
# 原代码
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner import load_checkpoint
from mmengine import MMLogger

# 替换为
import torch.nn as nn
from basicvsrpp.mmagic.pytorch_utils import constant_init, kaiming_init, load_checkpoint, get_logger

# 使用
BaseModule = nn.Module  # 或者直接用 nn.Module
logger = get_logger(__name__)
```

### 步骤 3: 修改类定义

```python
# 原代码
class BasicVSRPlusPlusNet(BaseModule):
    def __init__(self):
        super().__init__()

# 替换为
class BasicVSRPlusPlusNet(nn.Module):
    def __init__(self):
        super().__init__()
```

## 📝 核心文件修改清单

### 优先级 1 (核心推理)

- [ ] `basicvsrpp/mmagic/basicvsr_plusplus_net.py`
  - [ ] BaseModule → nn.Module (4 处)
  - [ ] constant_init (1 处)
  - [ ] kaiming_init (使用 pytorch_utils)
  - [ ] load_checkpoint (1 处)

- [ ] `basicvsrpp/mmagic/model_utils.py`
  - [ ] 所有 init 函数
  - [ ] _BatchNorm 检查

- [ ] `basicvsrpp/inference.py`
  - [ ] load_checkpoint (1 处)
  - [ ] Config → dict

### 优先级 2 (工具函数)

- [ ] `basicvsrpp/mmagic/perceptual_loss.py`
- [ ] `basicvsrpp/mmagic/real_basicvsr.py`
- [ ] `basicvsrpp/mmagic/unet_disc.py`

## 🧪 测试验证

### 快速测试脚本

```python
# test_migration.py
import torch
import torch.nn as nn

def test_basic_module():
    """测试基础模块"""
    from basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

    model = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=7, spynet_pretrained=None)
    assert isinstance(model, nn.Module)
    print("✅ BaseModule 替换成功")

def test_init_functions():
    """测试初始化函数"""
    from basicvsrpp.mmagic.pytorch_utils import constant_init, kaiming_init

    conv = nn.Conv2d(3, 64, 3)
    constant_init(conv, 1.0)
    assert torch.all(conv.weight == 1.0)
    print("✅ 初始化函数工作正常")

def test_inference():
    """测试推理"""
    from basicvsrpp.inference import get_default_gan_inference_config

    config = get_default_gan_inference_config()
    assert config['type'] == 'BasicVSRPlusPlusGan'
    print("✅ 推理接口正常")

if __name__ == '__main__':
    test_basic_module()
    test_init_functions()
    test_inference()
    print("\n🎉 所有测试通过！")
```

运行测试:
```bash
python test_migration.py
```

## 🔍 查找和替换命令

```bash
# 1. 查找所有 BaseModule 使用
grep -rn "BaseModule" basicvsrpp/mmagic/*.py

# 2. 查找所有 mmengine 导入
grep -rn "from mmengine" basicvsrpp/ --include="*.py"

# 3. 统计需要修改的地方
grep -r "from mmengine" basicvsrpp/ --include="*.py" | wc -l

# 4. 查找特定函数使用
grep -rn "constant_init\|kaiming_init" basicvsrpp/mmagic/

# 5. 查找 MMLogger 使用
grep -rn "MMLogger" basicvsrpp/
```

## ⚠️ 常见陷阱

### 1. 初始化函数参数差异

```python
# ❌ 错误：直接替换会丢失参数
kaiming_init(m, a=0, mode='fan_in', bias=0)
# 变成
init.kaiming_normal_(m.weight, a=0, mode='fan_in')  # ❌ 缺少 bias 处理

# ✅ 正确：使用包装函数
from basicvsrpp.mmagic.pytorch_utils import kaiming_init
kaiming_init(m, a=0, mode='fan_in', bias=0)  # ✅ 保持原有行为
```

### 2. Checkpoint 格式

```python
# ❌ 错误：假设固定键名
state_dict = checkpoint['state_dict']  # ❌ 可能不存在

# ✅ 正确：兼容多种格式
state_dict = checkpoint.get('state_dict',
             checkpoint.get('model', checkpoint))
```

### 3. BatchNorm 检查

```python
# ❌ 错误：使用 mmengine 的包装类
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

# ✅ 正确：使用 PyTorch 原生检查
def is_batch_norm(module):
    return isinstance(module, (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm
    ))
```

## 📦 修改后的依赖

```python
# requirements.txt (修改后)
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
av>=10.0.0
scipy>=1.10.0
# mmengine>=0.10.0  # ✅ 已移除
```

## 🎯 检查点清单

完成每一项后打钩：

### 阶段 1: 基础替换
- [ ] 创建 `pytorch_utils.py` 工具文件
- [ ] 替换 `basicvsr_plusplus_net.py` 中的 BaseModule
- [ ] 替换所有初始化函数调用
- [ ] 替换 load_checkpoint
- [ ] 替换 MMLogger
- [ ] 运行基础测试

### 阶段 2: 验证
- [ ] 测试模型创建
- [ ] 测试前向传播
- [ ] 测试 checkpoint 加载（如有）
- [ ] 对比性能（可选）

### 阶段 3: 清理
- [ ] 移除 mmengine 导入
- [ ] 更新 requirements.txt
- [ ] 更新文档
- [ ] 提交代码

## 💡 有用的提示

1. **渐进式修改**: 一次只修改一个文件，立即测试
2. **保留备份**: 修改前先 `git commit`
3. **使用 diff**: 对比修改前后的差异
4. **检查依赖**: 修改后运行 `pip check`

## 🆘 遇到问题？

### 问题 1: 导入错误
```
ImportError: cannot import name 'BaseModule' from 'mmengine.model'
```
**解决**: 确保已替换所有 BaseModule 为 nn.Module

### 问题 2: 初始化错误
```
TypeError: constant_init() missing 1 required positional argument: 'bias'
```
**解决**: 使用 pytorch_utils.py 中的包装函数

### 问题 3: Checkpoint 加载失败
```
KeyError: 'state_dict'
```
**解决**: 使用兼容多种格式的 load_checkpoint 函数

## 📚 参考资源

- 完整迁移计划: [MMENGINE_TO_PYTORCH_MIGRATION.md](MMENGINE_TO_PYTORCH_MIGRATION.md)
- PyTorch 文档: https://pytorch.org/docs/stable/
- PyTorch init 文档: https://pytorch.org/docs/stable/nn.init.html

---

**预计时间**: 2-3 天（核心功能）
**难度**: ⭐⭐⭐ 中等
**收益**: ⭐⭐⭐⭐⭐ 很高
