#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Generate the golden regression set (tests/golden/) from the CURRENT code.

Only run this on a known-good baseline. Regenerating after a refactor would
just bake the refactor's behavior into the reference and defeat the purpose.
"""

import datetime
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import golden_utils as gu
from basicvsrpp import register_all_modules
from basicvsrpp.inference import get_default_gan_inference_config, inference, load_model
from basicvsrpp.mmagic.basicvsr_plusplus_net import (BasicVSRPlusPlusNet, SPyNet,
                                                     SecondOrderDeformableAlignment)
from basicvsrpp.mmagic.registry import MODELS


def git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return os.environ.get('GITHUB_SHA', 'unknown')


def put(golden: dict, prefix: str, summary: dict) -> None:
    for k, v in summary.items():
        golden[f'{prefix}.{k}'] = v


def main() -> int:
    torch.manual_seed(0)
    register_all_modules()
    config = get_default_gan_inference_config()

    model = MODELS.build(config)
    param_keys = {k for k, _ in model.named_parameters()}
    full_sd = model.state_dict()
    manifest = {
        'baseline_commit': git_commit(),
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'versions': {'python': sys.version.split()[0], 'torch': torch.__version__,
                     'torchvision': torchvision.__version__, 'numpy': np.__version__},
        'state_dict': [{'key': k, 'shape': list(v.shape), 'dtype': str(v.dtype).replace('torch.', ''),
                        'kind': 'param' if k in param_keys else 'buffer'}
                       for k, v in full_sd.items()],
    }
    buffers = {e['key']: full_sd[e['key']].numpy() for e in manifest['state_dict'] if e['kind'] == 'buffer'}
    sd = gu.build_state_dict(manifest, buffers)

    golden = {f'buffer.{k}': v for k, v in buffers.items()}
    video = gu.synthetic_video()
    golden['input.video'] = video

    with torch.no_grad():
        frames = gu.video_to_tensor(video)
        small = F.interpolate(frames[0:2], scale_factor=0.25, mode='bicubic')
        spynet = SPyNet(pretrained=None)
        spynet.load_state_dict(gu.sub_state_dict(sd, 'generator.spynet.'), strict=True)
        spynet.eval()
        golden['spynet.ref'] = small[0:1].numpy()
        golden['spynet.supp'] = small[1:2].numpy()
        golden['spynet.flow'] = spynet(small[0:1], small[1:2]).numpy()

        align = SecondOrderDeformableAlignment(128, 64, 3, padding=1, deform_groups=16,
                                               max_residue_magnitude=10)
        align.load_state_dict(gu.sub_state_dict(sd, 'generator.deform_align.backward_1.'), strict=True)
        align.eval()
        golden['align.out'] = align(*gu.align_inputs()).numpy()

        net = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=15, max_residue_magnitude=10,
                                  spynet_pretrained=None)
        net.load_state_dict(gu.sub_state_dict(sd, 'generator.'), strict=True)
        net.eval()
        lqs = gu.net_input(video)
        net_out = net(lqs).numpy()
        put(golden, 'net.out', gu.summarize(net_out, (3, 4), 4))

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, 'golden.pth')
        torch.save({'meta': {'note': 'golden'}, 'state_dict': sd}, ckpt)
        loaded = load_model(config, ckpt, 'cpu')
        for k, v in loaded.state_dict().items():
            assert torch.equal(v, sd[k]), f'checkpoint did not round-trip for {k}'
        out_full = np.stack(inference(loaded, list(video), 'cpu'))
        out_chunk = np.stack(inference(loaded, list(video), 'cpu', max_frames=gu.CHUNK_FRAMES))
    put(golden, 'infer.full', gu.summarize(out_full, (1, 2), 2))
    put(golden, 'infer.chunk', gu.summarize(out_chunk, (1, 2), 4))

    os.makedirs(gu.GOLDEN_DIR, exist_ok=True)
    np.savez_compressed(gu.DATA_PATH, **golden)
    with open(gu.MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=1)
        f.write('\n')

    resid = net_out - lqs.numpy()
    print(f'baseline commit: {manifest["baseline_commit"]}')
    print(f'state_dict entries: {len(manifest["state_dict"])} ({len(buffers)} buffers)')
    print(f'spynet flow: mean|.|={np.abs(golden["spynet.flow"]).mean():.4f} max={np.abs(golden["spynet.flow"]).max():.4f}')
    print(f'align out: std={golden["align.out"].std():.4f}')
    print(f'net residual (out - in): std={resid.std():.4f} max|.|={np.abs(resid).max():.4f}')
    for name, out in (('full', out_full), ('chunk', out_chunk)):
        sat = np.mean((out == 0) | (out == 255))
        diff = np.abs(out.astype(int) - video.astype(int)).mean()
        print(f'inference {name}: saturated={sat:.3f} mean|out-in|={diff:.2f}')
    print(f'wrote {gu.DATA_PATH} ({os.path.getsize(gu.DATA_PATH) / 1e6:.2f} MB) and {gu.MANIFEST_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
