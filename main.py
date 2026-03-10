#!/usr/bin/env python3
"""
main.py — Entry point for Tech-App-Devoir-II (INF 6243).

Role:
  - Parse CLI arguments (e.g. --train, --evaluate, --data-path, --config).
  - Load configuration (from config.py or overrides).
  - Orchestrate the pipeline: data loading -> training and/or evaluation -> output.

Structure:
  1. Imports (standard library, then third-party, then local: config, data, train, evaluate).
  2. Argument parser (argparse): subcommands or flags for train/evaluate, paths, device.
  3. main():
     - Set device (CPU/CUDA from config or env).
     - If train: call train.run() or equivalent with config and data.
     - If evaluate: load model + data, call evaluate.run() and print/save metrics.
     - Optional: logging setup, seed for reproducibility.
  4. if __name__ == "__main__": main() or parser.invoke().
"""


def main():
    """Orchestrate pipeline based on CLI. Implement: parse args, load config, run train/evaluate."""
    pass


if __name__ == "__main__":
    main()
