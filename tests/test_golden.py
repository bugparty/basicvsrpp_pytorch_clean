#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Compare the current code against the golden set produced by generate_golden.py."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import golden_utils as gu
from basicvsrpp.inference import get_default_gan_inference_config, inference, load_model
from basicvsrpp.mmagic.basicvsr_plusplus_net import (BasicVSRPlusPlusNet, SPyNet,
                                                     SecondOrderDeformableAlignment)

FLOAT_TOL = dict(rtol=1e-4, atol=1e-4)
UINT8_MAX_DIFF = 1
UINT8_MAX_MISMATCH = 0.005

failures = []


def check_float(name, actual, expected):
    actual = np.asarray(actual)
    if actual.shape != expected.shape:
        failures.append(f'{name}: shape {actual.shape} != {expected.shape}')
        return
    if not np.allclose(actual, expected, **FLOAT_TOL):
        failures.append(f'{name}: max |diff| = {np.abs(actual - expected).max():.3e}')


def check_uint8(name, actual, expected):
    if actual.shape != expected.shape:
        failures.append(f'{name}: shape {actual.shape} != {expected.shape}')
        return
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    mismatch = np.mean(diff > 0)
    if diff.max() > UINT8_MAX_DIFF or mismatch > UINT8_MAX_MISMATCH:
        failures.append(f'{name}: max |diff| = {diff.max()}, mismatching = {mismatch:.4f}')


def check_summary(name, arr, golden, spatial_axes, stride):
    s = gu.summarize(arr, spatial_axes, stride)
    if arr.dtype == np.uint8:
        check_uint8(f'{name}.sub', s['sub'], golden[f'{name}.sub'])
        check_uint8(f'{name}.crop', s['crop'], golden[f'{name}.crop'])
        expected = golden[f'{name}.stats']
        if not (np.allclose(s['stats'][:2], expected[:2], atol=0.05)
                and np.allclose(s['stats'][2:], expected[2:], atol=UINT8_MAX_DIFF)):
            failures.append(f'{name}.stats: {s["stats"]} != {expected}')
    else:
        check_float(f'{name}.sub', s['sub'], golden[f'{name}.sub'])
        check_float(f'{name}.crop', s['crop'], golden[f'{name}.crop'])
        check_float(f'{name}.stats', s['stats'], golden[f'{name}.stats'])


def run(name, fn):
    before = len(failures)
    print(f'{name} ...', flush=True)
    fn()
    print(f'  {"OK" if len(failures) == before else "FAILED"}')


def main() -> int:
    if not os.path.exists(gu.DATA_PATH) or not os.path.exists(gu.MANIFEST_PATH):
        print(f'golden data missing ({gu.DATA_PATH}); generate it on a known-good baseline '
              'with tests/generate_golden.py')
        return 1

    torch.manual_seed(0)
    with open(gu.MANIFEST_PATH) as f:
        manifest = json.load(f)
    golden = dict(np.load(gu.DATA_PATH))
    buffers = {k[len('buffer.'):]: v for k, v in golden.items() if k.startswith('buffer.')}
    sd = gu.build_state_dict(manifest, buffers)
    video = golden['input.video']
    print(f'golden baseline: {manifest["baseline_commit"]} (torch {manifest["versions"]["torch"]}), '
          f'running with torch {torch.__version__}')

    def state_dict_layout():
        net = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=15, spynet_pretrained=None)
        current = {k: list(v.shape) for k, v in net.state_dict().items()}
        expected = {e['key'][len('generator.'):]: e['shape'] for e in manifest['state_dict']
                    if e['key'].startswith('generator.')}
        if current != expected:
            missing = sorted(set(expected) - set(current))[:5]
            extra = sorted(set(current) - set(expected))[:5]
            reshaped = sorted(k for k in set(current) & set(expected) if current[k] != expected[k])[:5]
            failures.append(f'generator state_dict layout changed: missing={missing} '
                            f'extra={extra} reshaped={reshaped}')

    def spynet_case():
        spynet = SPyNet(pretrained=None)
        spynet.load_state_dict(gu.sub_state_dict(sd, 'generator.spynet.'), strict=True)
        spynet.eval()
        with torch.no_grad():
            flow = spynet(torch.from_numpy(golden['spynet.ref']), torch.from_numpy(golden['spynet.supp']))
        check_float('spynet.flow', flow.numpy(), golden['spynet.flow'])

    def align_case():
        align = SecondOrderDeformableAlignment(128, 64, 3, padding=1, deform_groups=16,
                                               max_residue_magnitude=10)
        align.load_state_dict(gu.sub_state_dict(sd, 'generator.deform_align.backward_1.'), strict=True)
        align.eval()
        with torch.no_grad():
            out = align(*gu.align_inputs())
        check_float('align.out', out.numpy(), golden['align.out'])

    def net_case():
        net = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=15, max_residue_magnitude=10,
                                  spynet_pretrained=None)
        net.load_state_dict(gu.sub_state_dict(sd, 'generator.'), strict=True)
        net.eval()
        with torch.no_grad():
            out = net(gu.net_input(video)).numpy()
        check_summary('net.out', out, golden, (3, 4), 4)

    def load_and_infer_case():
        config = get_default_gan_inference_config()
        with tempfile.TemporaryDirectory() as tmp:
            # Real training checkpoints also carry discriminator weights; they must be ignored.
            wrapped = dict(sd)
            wrapped['discriminator.conv0.weight'] = torch.zeros(4, 3, 3, 3)
            paths = {}
            for fmt, obj in (('wrapped', {'meta': {'note': 'golden'}, 'state_dict': wrapped}),
                             ('raw', dict(sd)),
                             ('ddp_prefix', {'state_dict': {f'module.{k}': v for k, v in sd.items()}})):
                paths[fmt] = os.path.join(tmp, f'{fmt}.pth')
                torch.save(obj, paths[fmt])

            model = load_model(config, paths['wrapped'], 'cpu')
            if model.training:
                failures.append('load_model returned a model in training mode')
            out_full = np.stack(inference(model, list(video), 'cpu'))
            out_chunk = np.stack(inference(model, list(video), 'cpu', max_frames=gu.CHUNK_FRAMES))
            check_summary('infer.full', out_full, golden, (1, 2), 2)
            check_summary('infer.chunk', out_chunk, golden, (1, 2), 4)

            reference = model.state_dict()
            for fmt in ('raw', 'ddp_prefix'):
                other = load_model(config, paths[fmt], 'cpu').state_dict()
                if other.keys() != reference.keys() or not all(torch.equal(other[k], reference[k]) for k in reference):
                    failures.append(f'load_model({fmt} checkpoint) loads different weights than the wrapped format')

    run('generator state_dict layout', state_dict_layout)
    run('SPyNet flow', spynet_case)
    run('SecondOrderDeformableAlignment', align_case)
    run('BasicVSRPlusPlusNet forward', net_case)
    run('load_model + inference', load_and_infer_case)

    if failures:
        print('\nGolden regression FAILED:')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('\nAll golden checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
