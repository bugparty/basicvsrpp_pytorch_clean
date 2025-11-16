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
└── lib/                 # 辅助工具库
    ├── image_utils.py
    ├── random_utils.py
    ├── transforms.py
    ├── mosaic_utils.py
    ├── degradations.py
    └── ...
```

## 主要特性

- **无 mmcv 依赖**: 这个版本移除了对 mmcv 的复杂依赖
- **独立运行**: 包含所有必要的工具函数
- **GAN 版本**: 包含 BasicVSR++ GAN 实现用于视频超分辨率
- **推理接口**: 提供简单的推理接口

## 使用示例

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

## 依赖说明

基本 PyTorch 依赖，详见 `requirements.txt`

## 许可证

- SPDX-FileCopyrightText: Lada Authors
- SPDX-License-Identifier: AGPL-3.0

## 来源

提取自: https://github.com/ladaapp/lada/tree/main/lada/basicvsrpp
