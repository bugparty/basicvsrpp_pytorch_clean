# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

# The networks live in basicvsrpp.models; this module registers them for mmengine training configs.

from basicvsrpp.models.basicvsr_plusplus_net import *  # noqa: F401,F403
from basicvsrpp.models.basicvsr_plusplus_net import BasicVSRPlusPlusNet
from .registry import MODELS

MODELS.register_module(module=BasicVSRPlusPlusNet)
