# BasicVSR++ Demo

这个 demo 展示了如何使用干净版的 BasicVSR++ 进行视频修复。

## 使用方法

### 基本用法

```bash
python demo/restoration_video_demo.py <input> <output> <checkpoint>
```

参数说明：
- `input`: 输入视频文件或包含帧的目录
- `output`: 输出视频文件或输出帧的目录
- `checkpoint`: 模型权重文件路径

### 示例

**1. 视频输入 → 视频输出**
```bash
python demo/restoration_video_demo.py input.mp4 output.mp4 checkpoint.pth
```

**2. 图像序列输入 → 图像序列输出**
```bash
python demo/restoration_video_demo.py input_frames/ output_frames/ checkpoint.pth
```

**3. 视频输入 → 图像序列输出**
```bash
python demo/restoration_video_demo.py input.mp4 output_frames/ checkpoint.pth
```

### 可选参数

```bash
python demo/restoration_video_demo.py input.mp4 output.mp4 checkpoint.pth \
    --config config.py \           # 自定义配置文件
    --device cuda:0 \              # 使用的设备 (cuda:0, cpu 等)
    --max-frames 30 \              # 一次处理的最大帧数
    --fps 30.0                     # 输出视频的 FPS
```

## 参数详解

- `--config`: 可选的配置文件路径。如果不提供，将使用默认的 GAN 配置
- `--device`: 运行设备，默认 `cuda:0`。如果没有 GPU 可以使用 `cpu`
- `--max-frames`: 一次处理的最大帧数。用于控制显存使用。-1 表示一次处理所有帧（默认）
- `--fps`: 输出视频的帧率，默认 25.0。如果输入是视频，会自动使用输入视频的帧率

## 注意事项

1. **输入格式**: 支持常见视频格式 (.mp4, .mov, .avi, .mkv) 和图像序列 (.png, .jpg)
2. **显存管理**: 如果显存不足，使用 `--max-frames` 参数分批处理
3. **输出质量**: 建议使用图像序列作为输出以获得最佳质量（视频会有压缩损失）
4. **模型权重**: 需要提供训练好的 BasicVSR++ 模型权重文件

## 与原版 demo 的区别

相比原版 BasicVSR++ 的 demo，这个版本：

- ✅ **无需 mmcv/mmagic**: 不依赖复杂的 mmcv 和 mmagic 框架
- ✅ **更简洁**: 代码更简单直观，易于理解和修改
- ✅ **独立运行**: 只需要基本的 PyTorch 和 OpenCV
- ✅ **保持兼容**: 核心功能与原版一致

## 获取模型权重

BasicVSR++ 的预训练权重可以从以下来源获取：
- [官方 BasicVSR++ 仓库](https://github.com/ckkelvinchan/BasicVSR_PlusPlus)
- [MMagic Model Zoo](https://github.com/open-mmlab/mmagic/tree/main/configs/basicvsr_pp)

注意：需要确保权重文件与我们的模型架构兼容。
