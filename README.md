# BasicVSR++ (Clean Version)

这是从 [lada](https://github.com/ladaapp/lada) 项目中提取的干净版本的 BasicVSR++，移除了复杂的 mmcv 依赖。

## 项目结构

```
.
├── basicvsrpp/           # BasicVSR++ 核心代码
│   ├── mmagic/          # 核心模块和工具
│   ├── __init__.py
│   ├── basicvsrpp_gan.py
│   ├── deformconv.py
│   ├── inference.py
│   └── mosaic_video_dataset.py
├── lib/                 # 辅助工具库
│   ├── image_utils.py
│   ├── random_utils.py
│   ├── transforms.py
│   ├── mosaic_utils.py
│   ├── degradations.py
│   └── ...
├── configs/             # 训练配置文件
│   ├── train_basicvsrpp_example.py
│   ├── train_minimal.py
│   └── README.md
└── tools/               # 训练脚本
    └── train.py
```

## 主要特性

- **无 mmcv 依赖**: 这个版本移除了对 mmcv 的复杂依赖，只使用轻量级的 mmengine
- **独立运行**: 包含所有必要的工具函数
- **GAN 版本**: 包含 BasicVSR++ GAN 实现用于视频超分辨率
- **推理接口**: 提供简单的推理接口
- **完整训练支持**: 包含训练脚本和配置文件，支持单卡和多卡训练

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 推理示例

```python
from basicvsrpp.inference import load_model, inference, get_default_gan_inference_config
import torch

# 加载模型
config = get_default_gan_inference_config()
model = load_model(config, "path/to/checkpoint.pth", device="cuda:0")

# 推理
# video 是一个包含多帧图像的列表 (numpy arrays)
result = inference(model, video, device="cuda:0")
```

### 训练示例

#### 1. 准备数据集

准备视频数据集的元数据文件（JSON格式），存放在指定目录下。详见 `configs/README.md`。

#### 2. 修改配置文件

```bash
# 编辑配置文件，修改数据集路径
vim configs/train_minimal.py

# 主要修改以下参数：
# - metadata_root_dir: 数据集元数据路径
# - batch_size: 根据你的GPU内存调整
# - num_workers: 数据加载线程数
```

#### 3. 开始训练

```bash
# 单卡训练
python tools/train.py configs/train_minimal.py

# 多卡训练（例如4卡）
torchrun --nproc_per_node=4 tools/train.py configs/train_basicvsrpp_example.py

# 恢复训练
python tools/train.py configs/train_minimal.py --resume work_dirs/train_minimal/latest.pth

# 使用混合精度训练（更快，更省显存）
python tools/train.py configs/train_minimal.py --amp
```

#### 4. 查看训练结果

```bash
# 训练日志和检查点保存在 work_dir 目录
ls work_dirs/train_minimal/

# 查看训练日志
cat work_dirs/train_minimal/*.log
```

更多训练选项和配置说明，请参考 `configs/README.md`。

## 依赖说明

基本 PyTorch 依赖，详见 `requirements.txt`。

主要依赖：
- `torch>=2.0.0` - PyTorch 深度学习框架
- `mmengine>=0.10.0` - 轻量级训练框架（无需 mmcv）
- `torchvision>=0.15.0` - 提供 deformable convolution
- `opencv-python>=4.8.0` - 图像处理
- `av>=10.0.0` - 视频处理

## 与原版的区别

| 特性 | 原版 BasicVSR++ | 本项目 |
|------|----------------|--------|
| **依赖** | 需要 mmcv-full (编译复杂) | 只需 mmengine (纯Python) |
| **Deformable Conv** | mmcv.ops | torchvision.ops |
| **安装难度** | 困难 (需要CUDA编译) | 简单 (pip install) |
| **代码结构** | 分散在 MMEditing | 独立完整的代码库 |
| **训练** | 完整的 MMEditing 框架 | 简化的训练脚本 |
| **推理** | 需要完整框架 | 独立推理接口 |

## 许可证

- SPDX-FileCopyrightText: Lada Authors
- SPDX-License-Identifier: AGPL-3.0

## 来源

提取自: https://github.com/ladaapp/lada/tree/main/lada/basicvsrpp
