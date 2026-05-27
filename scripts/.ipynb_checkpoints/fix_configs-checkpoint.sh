#!/bin/bash
set -e

ORIGINAL=/workdir/models/Qwen3.5-2B-Base
python3 fix_config.py --original "$ORIGINAL" --adapted /workdir/models/RuadaptQwen3.5-2B-Base-u128_trimmed_u128_smart_k100_sub_targeted_fr0.3_ps0.5_3e_wsd0.6_lr3e4