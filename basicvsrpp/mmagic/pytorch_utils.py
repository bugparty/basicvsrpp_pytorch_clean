"""
PyTorch 原生工具函数

这个模块提供了 mmengine 常用功能的 PyTorch 原生实现:
- 权重初始化函数
- Checkpoint 加载/保存
- 日志工具
- 其他辅助函数

作者: BasicVSR++ Clean 项目
许可: AGPL-3.0
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import logging
import sys
from typing import Optional, Dict, Any, Union
from pathlib import Path


# ============================================================================
# 权重初始化函数
# ============================================================================

def constant_init(module: Union[nn.Module, torch.Tensor], val: float, bias: float = 0) -> None:
    """常量初始化

    Args:
        module: nn.Module 或 torch.Tensor
        val: 权重初始化值
        bias: bias 初始化值

    Examples:
        >>> conv = nn.Conv2d(3, 64, 3)
        >>> constant_init(conv, 1.0, 0.0)
    """
    if isinstance(module, nn.Module):
        if hasattr(module, 'weight') and module.weight is not None:
            init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            init.constant_(module.bias, bias)
    else:
        # 直接是张量
        init.constant_(module, val)


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu',
    bias: float = 0,
    distribution: str = 'normal'
) -> None:
    """Kaiming 初始化

    Args:
        module: nn.Module
        a: 激活函数的负斜率 (仅用于 leaky_relu)
        mode: 'fan_in' 或 'fan_out'
        nonlinearity: 激活函数名称
        bias: bias 初始化值
        distribution: 'normal' 或 'uniform'

    Examples:
        >>> conv = nn.Conv2d(3, 64, 3)
        >>> kaiming_init(conv, a=0, mode='fan_in')
    """
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)

    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


def xavier_init(
    module: nn.Module,
    gain: float = 1,
    bias: float = 0,
    distribution: str = 'normal'
) -> None:
    """Xavier 初始化

    Args:
        module: nn.Module
        gain: 增益系数
        bias: bias 初始化值
        distribution: 'normal' 或 'uniform'

    Examples:
        >>> linear = nn.Linear(128, 64)
        >>> xavier_init(linear, gain=1.0)
    """
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            init.xavier_uniform_(module.weight, gain=gain)
        else:
            init.xavier_normal_(module.weight, gain=gain)

    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


def normal_init(
    module: nn.Module,
    mean: float = 0,
    std: float = 1,
    bias: float = 0
) -> None:
    """正态分布初始化

    Args:
        module: nn.Module
        mean: 均值
        std: 标准差
        bias: bias 初始化值

    Examples:
        >>> conv = nn.Conv2d(3, 64, 3)
        >>> normal_init(conv, mean=0, std=0.01)
    """
    if hasattr(module, 'weight') and module.weight is not None:
        init.normal_(module.weight, mean, std)

    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


def default_init_weights(module: nn.Module, scale: float = 1) -> None:
    """默认权重初始化

    为 Conv2d, Linear, BatchNorm 等常用层提供默认初始化。

    Args:
        module: nn.Module
        scale: 缩放因子

    Examples:
        >>> model = MyModel()
        >>> default_init_weights(model, scale=0.1)
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            constant_init(m, val=1, bias=0)


# ============================================================================
# Checkpoint 加载/保存
# ============================================================================

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    map_location: str = 'cpu',
    strict: bool = True,
    logger: Optional[logging.Logger] = None
) -> nn.Module:
    """加载模型 checkpoint

    支持多种 checkpoint 格式:
    - 直接的 state_dict
    - 包含 'state_dict' 键的字典
    - 包含 'model' 键的字典
    - 包含 'generator' 键的字典（GAN 模型）

    自动处理 DataParallel/DistributedDataParallel 的 'module.' 前缀。

    Args:
        model: 要加载权重的模型
        checkpoint_path: checkpoint 文件路径
        map_location: 加载到的设备 ('cpu', 'cuda', 'cuda:0' 等)
        strict: 是否严格匹配 state_dict 的键
        logger: 日志记录器

    Returns:
        加载权重后的模型

    Examples:
        >>> model = MyModel()
        >>> model = load_checkpoint(model, 'checkpoint.pth', map_location='cuda:0')
    """
    if logger is None:
        logger = get_logger(__name__)

    checkpoint_path = str(checkpoint_path)
    logger.info(f'Loading checkpoint from {checkpoint_path}')

    # 加载 checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    except Exception as e:
        logger.error(f'Failed to load checkpoint: {e}')
        raise

    # 提取 state_dict（支持多种格式）
    if isinstance(checkpoint, dict):
        # 尝试多个可能的键名
        for key in ['state_dict', 'model', 'generator', 'state']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                logger.info(f'Found state_dict in key: {key}')
                break
        else:
            # 没有找到标准键，假设整个 checkpoint 就是 state_dict
            state_dict = checkpoint
            logger.info('Using checkpoint as state_dict directly')
    else:
        state_dict = checkpoint
        logger.info('Checkpoint is a direct state_dict')

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
        logger.warning(f'Missing keys in checkpoint ({len(missing_keys)}): {missing_keys[:5]}...')
    if unexpected_keys:
        logger.warning(f'Unexpected keys in checkpoint ({len(unexpected_keys)}): {unexpected_keys[:5]}...')

    if not missing_keys and not unexpected_keys:
        logger.info('✅ Checkpoint loaded successfully!')
    else:
        logger.warning(f'⚠️  Checkpoint loaded with warnings (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})')

    return model


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    meta: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """保存模型 checkpoint

    Args:
        model: 要保存的模型
        checkpoint_path: 保存路径
        meta: 额外的元信息（如 epoch, optimizer, loss 等）
        logger: 日志记录器

    Examples:
        >>> model = MyModel()
        >>> meta = {'epoch': 100, 'loss': 0.5}
        >>> save_checkpoint(model, 'checkpoint.pth', meta=meta)
    """
    if logger is None:
        logger = get_logger(__name__)

    checkpoint_path = str(checkpoint_path)

    # 构建 checkpoint
    checkpoint = {
        'state_dict': model.state_dict()
    }

    # 添加元信息
    if meta is not None:
        checkpoint.update(meta)

    # 保存
    try:
        # 确保目录存在
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f'✅ Checkpoint saved to {checkpoint_path}')
    except Exception as e:
        logger.error(f'Failed to save checkpoint: {e}')
        raise


# ============================================================================
# 日志工具
# ============================================================================

def setup_logger(
    name: str = __name__,
    log_file: Optional[str] = None,
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

    Examples:
        >>> logger = setup_logger('my_module', log_file='train.log')
        >>> logger.info('Training started')
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

    如果 logger 不存在，使用默认配置创建。

    Args:
        name: logger 名称

    Returns:
        logger 实例

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info('Hello')
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 如果没有配置过，使用默认配置
        return setup_logger(name)
    return logger


# ============================================================================
# 其他工具函数
# ============================================================================

def get_module_device(module: nn.Module) -> torch.device:
    """获取模块所在的设备

    Args:
        module: nn.Module

    Returns:
        设备 (torch.device)

    Examples:
        >>> model = MyModel().cuda()
        >>> device = get_module_device(model)
        >>> print(device)  # cuda:0
    """
    try:
        param = next(module.parameters())
        return param.device
    except StopIteration:
        raise ValueError('The input module should contain parameters.')


def set_requires_grad(nets: Union[nn.Module, list], requires_grad: bool = False) -> None:
    """设置网络的 requires_grad 属性

    Args:
        nets: 单个网络或网络列表
        requires_grad: 是否需要梯度

    Examples:
        >>> discriminator = MyDiscriminator()
        >>> set_requires_grad(discriminator, False)  # 冻结判别器
    """
    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def is_batch_norm(module: nn.Module) -> bool:
    """检查模块是否为 BatchNorm 层

    Args:
        module: nn.Module

    Returns:
        是否为 BatchNorm

    Examples:
        >>> bn = nn.BatchNorm2d(64)
        >>> is_batch_norm(bn)  # True
    """
    return isinstance(module, (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    ))


# ============================================================================
# 向后兼容别名
# ============================================================================

# 为了兼容旧代码，提供一些别名
update_init_info = lambda *args, **kwargs: None  # mmengine 中用于记录初始化信息，这里忽略


__all__ = [
    # 初始化函数
    'constant_init',
    'kaiming_init',
    'xavier_init',
    'normal_init',
    'default_init_weights',
    # Checkpoint
    'load_checkpoint',
    'save_checkpoint',
    # 日志
    'setup_logger',
    'get_logger',
    # 工具
    'get_module_device',
    'set_requires_grad',
    'is_batch_norm',
]
