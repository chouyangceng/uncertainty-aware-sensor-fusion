from __future__ import annotations

import argparse

from uncertainty_sensor_fusion.experiments import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sensor fusion reliability experiments")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", default="artifacts/sensor-fusion")
    args = parser.parse_args()
    print(run(args.seed, args.frames, args.output))


if __name__ == "__main__":
    main()
