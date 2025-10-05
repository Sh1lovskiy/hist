"""CLI entry point for cross-validated MIL / GNN training."""
from __future__ import annotations

import random

import numpy as np
import torch

from config import build_config, build_parser
from log_setup import configure_logging
from runner import CrossValRunner


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = build_config(args)
    cfg.output.mkdir(parents=True, exist_ok=True)
    configure_logging(cfg.output, cfg.verbose)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    runner = CrossValRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
