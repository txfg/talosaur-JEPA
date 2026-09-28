#!/usr/bin/env python
"""JEPA pretraining entry point (Hydra). See configs/pretrain.yaml and README "M2".

python scripts/train.py experiment=debug_cpu
torchrun --standalone --nproc_per_node=4 scripts/train.py experiment=ijepa_tiny_224 degrade=context_only
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from talosaur.ssl.engine import Trainer
    from talosaur.utils import dist

    try:
        Trainer(cfg).train()
    finally:
        dist.cleanup()


if __name__ == "__main__":
    main()
