#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""The inference path must work without mmengine. Run as its own process."""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import basicvsrpp.checkpoint as ckpt
import basicvsrpp.inference as inf
import basicvsrpp.models  # noqa: F401


def test_no_mmengine_import():
    loaded = sorted(m for m in sys.modules if m == 'mmengine' or m.startswith('mmengine.'))
    assert not loaded, f'inference path imported {loaded[:3]}'


def test_training_checkpoint_with_unavailable_classes(tmp):
    # Mimic an mmengine training checkpoint: weights plus pickled objects of a package
    # that is not installed at load time.
    fake = types.ModuleType('fake_trainer_pkg')

    class HistoryBuffer:
        def __init__(self):
            self.values = [1.0, 2.0]

    HistoryBuffer.__module__ = 'fake_trainer_pkg'
    HistoryBuffer.__qualname__ = 'HistoryBuffer'
    fake.HistoryBuffer = HistoryBuffer
    sys.modules['fake_trainer_pkg'] = fake
    weights = {'module.generator.w': torch.arange(4.0)}
    path = os.path.join(tmp, 'train.pth')
    torch.save({'meta': {'iter': np.int64(3)}, 'state_dict': weights,
                'message_hub': {'log': HistoryBuffer()}}, path)
    del sys.modules['fake_trainer_pkg']

    sd = ckpt.load_state_dict(path)
    assert list(sd) == ['generator.w'] and torch.equal(sd['generator.w'], torch.arange(4.0))


def test_load_model_errors(tmp):
    config = inf.get_default_gan_inference_config()
    reference = inf.build_inference_model(config).state_dict()

    no_ema = os.path.join(tmp, 'no_ema.pth')
    torch.save({'state_dict': {k.replace('generator_ema.', 'generator.', 1): v for k, v in reference.items()}}, no_ema)
    try:
        inf.load_model(config, no_ema, 'cpu')
        raise AssertionError('expected KeyError for a checkpoint without generator_ema weights')
    except KeyError:
        pass

    bad_shape = dict(reference)
    key = next(k for k in bad_shape if k.endswith('conv_last.weight'))
    bad_shape[key] = torch.zeros(1)
    path = os.path.join(tmp, 'bad_shape.pth')
    torch.save(bad_shape, path)
    try:
        inf.load_model(config, path, 'cpu')
        raise AssertionError('expected RuntimeError for mismatched weights')
    except RuntimeError:
        pass

    try:
        inf.build_inference_model(dict(config, type='SomethingElse'))
        raise AssertionError('expected ValueError for an unknown model type')
    except ValueError:
        pass


def test_non_ema_model_uses_generator_weights(tmp):
    config = dict(type='BasicVSR', generator=dict(type='BasicVSRPlusPlusNet', mid_channels=64, num_blocks=15))
    sd = inf.build_inference_model(config).state_dict()
    assert all(k.startswith('generator.') for k in sd)
    path = os.path.join(tmp, 'basicvsr.pth')
    torch.save({'state_dict': sd}, path)
    model = inf.load_model(config, path, 'cpu')
    assert not model.training
    assert all(torch.equal(v, sd[k]) for k, v in model.state_dict().items())


def main() -> int:
    test_no_mmengine_import()
    print('  ✓ inference path does not import mmengine')
    with tempfile.TemporaryDirectory() as tmp:
        test_training_checkpoint_with_unavailable_classes(tmp)
        print('  ✓ training checkpoints with unavailable classes load')
        test_load_model_errors(tmp)
        print('  ✓ load_model rejects missing EMA weights, bad shapes and unknown types')
        test_non_ema_model_uses_generator_weights(tmp)
        print('  ✓ non-EMA models load generator weights')
    print('\n✓ All standalone inference tests passed!')
    return 0


if __name__ == '__main__':
    sys.exit(main())
