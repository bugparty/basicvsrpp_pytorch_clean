#!/usr/bin/env python3
# One-off check that test_golden.py fails for typical refactor mistakes (patches baseline internals).

import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn.functional as F
import torchvision

import test_golden
import basicvsrpp.inference as inf
import basicvsrpp.mmagic.basicvsr_plusplus_net as net_mod
from basicvsrpp.mmagic.real_basicvsr import RealBasicVSR
from lib.image_utils import img2tensor

orig_interpolate = F.interpolate
orig_deform = torchvision.ops.deform_conv2d


def bicubic_to_bilinear(*args, **kwargs):
    if kwargs.get('mode') == 'bicubic':
        kwargs['mode'] = 'bilinear'
    return orig_interpolate(*args, **kwargs)


MUTATIONS = {
    'inference uses generator instead of generator_ema':
        (RealBasicVSR, 'forward_tensor', lambda self, inputs, data_samples=None, training=False: self.generator(inputs)),
    'flow_warp align_corners=False':
        (net_mod, 'flow_warp', functools.partial(net_mod.flow_warp, align_corners=False)),
    'flow_warp nearest interpolation':
        (net_mod, 'flow_warp', functools.partial(net_mod.flow_warp, interpolation='nearest')),
    'bicubic downsample replaced by bilinear':
        (F, 'interpolate', bicubic_to_bilinear),
    'deformable conv without modulation mask':
        (torchvision.ops, 'deform_conv2d', lambda *a: orig_deform(*a[:7], None)),
    'forward/backward flows swapped':
        (net_mod.BasicVSRPlusPlusNet, 'compute_flow',
         lambda self, lqs, _orig=net_mod.BasicVSRPlusPlusNet.compute_flow: _orig(self, lqs)[::-1]),
    'BGR/RGB swapped on input':
        (inf, 'img2tensor', lambda imgs, bgr2rgb=False, float32=True: img2tensor(imgs, bgr2rgb=True, float32=float32)),
}


def main() -> int:
    test_golden.failures.clear()
    if test_golden.main() != 0:
        print('unmodified code does not pass the golden test')
        return 1
    missed = []
    for name, (owner, attr, replacement) in MUTATIONS.items():
        print(f'\n=== mutation: {name}')
        original = getattr(owner, attr)
        setattr(owner, attr, replacement)
        try:
            test_golden.failures.clear()
            detected = test_golden.main() != 0
        finally:
            setattr(owner, attr, original)
        verdict = 'DETECTED' if detected else 'MISSED'
        detail = '; '.join(test_golden.failures)[:400]
        print(f'=== {verdict}: {name}')
        print(f'::notice title=golden mutation::{verdict}: {name} | {detail}')
        if not detected:
            missed.append(name)
    print(f'\nmissed mutations: {missed}' if missed else '\nall mutations detected')
    return 1 if missed else 0


if __name__ == '__main__':
    sys.exit(main())
