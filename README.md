# BasicVSR++ (Clean Version)

This is a clean version of BasicVSR++ extracted from the [lada](https://github.com/ladaapp/lada) project, with complex mmcv dependencies removed.

## Project Structure

```
.
├── basicvsrpp/           # BasicVSR++ core code
│   ├── mmagic/          # Core modules and utilities
│   ├── __init__.py
│   ├── basicvsrpp_gan.py
│   ├── deformconv.py
│   ├── inference.py
│   └── mosaic_video_dataset.py
└── lib/                 # Utility library
    ├── image_utils.py
    ├── random_utils.py
    ├── transforms.py
    ├── mosaic_utils.py
    ├── degradations.py
    └── ...
```

## Key Features

- **No mmcv dependency**: This version removes the complex mmcv dependency
- **Standalone**: Includes all necessary utility functions
- **GAN version**: Includes BasicVSR++ GAN implementation for video super-resolution
- **Inference interface**: Provides a simple inference interface

## Usage Example

```python
from basicvsrpp.inference import load_model, inference, get_default_gan_inference_config
import torch

# Load model
config = get_default_gan_inference_config()
model = load_model(config, "path/to/checkpoint.pth", device="cuda:0")

# Inference
# video is a list of image frames (numpy arrays)
result = inference(model, video, device="cuda:0")
```

## Dependencies

Basic PyTorch dependencies, see `requirements.txt` for details

## License

- SPDX-FileCopyrightText: Lada Authors
- SPDX-License-Identifier: AGPL-3.0

## Source

Extracted from: https://github.com/ladaapp/lada/tree/main/lada/basicvsrpp
