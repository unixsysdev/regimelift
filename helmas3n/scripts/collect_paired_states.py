#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.src.gemma.state_extract import collect_paired_states, load_extraction_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect paired low/full regime states for HeLMAS-3n")
    parser.add_argument("--config", type=str, required=True, help="Path to extraction YAML config")
    args = parser.parse_args()

    cfg = load_extraction_config(args.config)
    manifest = collect_paired_states(cfg)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
