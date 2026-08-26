"""Evaluate whether reported 2-D fusion covariance matches empirical error."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from uncertainty_sensor_fusion.metrics.calibration import (
    calibration_report,
    normalized_estimation_error_squared,
)


def run_study(
    *,
    output_dir: str | Path = "artifacts/uncertainty-calibration",
    seed: int = 7,
    fast: bool = False,
    overwrite: bool = False,
) -> Path:
    """Generate controlled covariance failures, metrics and diagnostic plots."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    samples = 3000 if fast else 20000
    errors, true_covariances = _simulate_errors(samples, seed)
    reported = {
        "calibrated": true_covariances,
        "underreported_40pct": true_covariances * 0.4,
        "conservative_200pct": true_covariances * 2.0,
        "correlation_ignored": _diagonal_only(true_covariances),
    }
    reports = {
        name: calibration_report(errors, covariances)
        for name, covariances in reported.items()
    }
    metadata = {
        "seed": seed,
        "fast": fast,
        "samples": samples,
        "dimensions": 2,
        "expected_mean_nees": 2.0,
        "research_boundary": "heteroscedastic Gaussian simulation; not road-test evidence",
    }
    (output / "calibration.json").write_text(
        json.dumps(
            {"metadata": metadata, "scenarios": reports},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_coverage(output / "coverage.csv", reports)
    _write_calibration_curve(output / "calibration_curve.png", reports)
    _write_nees_distribution(output / "nees_distribution.png", errors, reported)
    _write_whitened_residuals(output / "whitened_residuals.png", errors, reported)
    _write_summary(output / "summary.md", metadata, reports)
    return output


def _simulate_errors(samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ranges = rng.uniform(5.0, 80.0, samples)
    headings = rng.uniform(-np.pi, np.pi, samples)
    covariances = np.empty((samples, 2, 2), dtype=float)
    errors = np.empty((samples, 2), dtype=float)
    for index, (distance, heading) in enumerate(zip(ranges, headings, strict=True)):
        longitudinal = 0.12 + 0.006 * distance
        lateral = 0.08 + 0.003 * distance
        rotation = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        covariance = rotation @ np.diag([longitudinal**2, lateral**2]) @ rotation.T
        covariances[index] = covariance
        errors[index] = np.linalg.cholesky(covariance) @ rng.normal(size=2)
    return errors, covariances


def _diagonal_only(covariances: np.ndarray) -> np.ndarray:
    result = np.zeros_like(covariances)
    result[:, 0, 0] = covariances[:, 0, 0]
    result[:, 1, 1] = covariances[:, 1, 1]
    return result


def _write_coverage(path: Path, reports: dict[str, dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario", "nominal_confidence", "empirical_coverage", "coverage_gap"])
        for name, report in reports.items():
            for nominal, empirical, gap in zip(
                report["confidence_levels"],
                report["empirical_coverage"],
                report["coverage_gap"],
                strict=True,
            ):
                writer.writerow([name, nominal, empirical, gap])


def _write_calibration_curve(path: Path, reports: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6c757d", label="ideal")
    markers = ("o", "s", "^", "D")
    for marker, (name, report) in zip(markers, reports.items(), strict=True):
        axis.plot(
            report["confidence_levels"],
            report["empirical_coverage"],
            marker=marker,
            linewidth=2,
            label=name,
        )
    axis.set(xlim=(0.45, 1.0), ylim=(0.2, 1.0))
    axis.set_xlabel("nominal confidence ellipse")
    axis.set_ylabel("empirical target coverage")
    axis.set_title("Position covariance calibration curve")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_nees_distribution(
    path: Path,
    errors: np.ndarray,
    reported: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.8, 5.8))
    limit = 14.0
    x_values = np.linspace(0.0, limit, 300)
    axis.plot(x_values, 1.0 - np.exp(-x_values / 2.0), "k--", label="ideal χ²(2)")
    for name, covariances in reported.items():
        nees = np.sort(normalized_estimation_error_squared(errors, covariances))
        empirical = np.arange(1, nees.size + 1) / nees.size
        visible = nees <= limit
        axis.plot(nees[visible], empirical[visible], linewidth=1.8, label=name)
    axis.set_xlabel("NEES")
    axis.set_ylabel("empirical cumulative probability")
    axis.set_title("NEES distribution versus χ² consistency target")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_whitened_residuals(
    path: Path,
    errors: np.ndarray,
    reported: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(9, 8.5), sharex=True, sharey=True)
    angle = np.linspace(0, 2 * np.pi, 200)
    radius_95 = np.sqrt(-2.0 * np.log(0.05))
    for axis, (name, covariances) in zip(axes.flat, reported.items(), strict=True):
        cholesky = np.linalg.cholesky(covariances)
        whitened = np.linalg.solve(cholesky, errors[..., None])[..., 0]
        axis.scatter(whitened[:500, 0], whitened[:500, 1], s=8, alpha=0.28)
        axis.plot(radius_95 * np.cos(angle), radius_95 * np.sin(angle), color="#e76f51")
        axis.set_title(name)
        axis.set_aspect("equal")
        axis.grid(alpha=0.18)
    figure.supxlabel("whitened x residual")
    figure.supylabel("whitened y residual")
    figure.suptitle("Reported 95% ellipse in whitened residual space")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_summary(
    path: Path,
    metadata: dict[str, Any],
    reports: dict[str, dict[str, Any]],
) -> None:
    rows = [
        f"| {name} | {report['mean_nees']:.3f} | {report['empirical_coverage'][3]:.1%} | {report['calibration_error']:.3f} | {report['classification']} |"
        for name, report in reports.items()
    ]
    text = [
        "# 二维位置协方差校准研究",
        "",
        "## 主要发现",
        "",
        "- 校准场景的平均 NEES 应接近二维状态自由度 2，名义 95% 椭圆的经验覆盖率也应接近 95%。",
        "- 低报协方差会缩小置信椭圆、提高 NEES，并产生危险的过度自信；保守协方差则降低信息利用率。",
        "- 只看 RMSE 无法区分这三种情况，因此跟踪精度必须和协方差一致性一起报告。",
        "",
        "| 场景 | 平均 NEES | 95% 覆盖率 | 校准误差 | 判定 |",
        "|---|---:|---:|---:|---|",
        *rows,
        "",
        "## 可复现设置",
        "",
        f"- 种子：`{metadata['seed']}`；样本数：{metadata['samples']}；误差维度：2。",
        "- 距离与朝向改变真实异方差协方差；各场景复用同一批误差，只改变报告协方差，形成受控比较。",
        "",
        "## 研究边界",
        "",
        "- 实验假设零均值高斯位置误差，不能覆盖多峰关联错误、目标漏检或非高斯遮挡尾部。",
        "- 结果用于说明校准方法，不是对真实相机、LiDAR、Radar 的道路性能声明。",
        "- 下一步应按距离、天气、目标类别和传感器健康状态分层，在真实标注数据上报告覆盖率置信区间。",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/uncertainty-calibration")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    return run_study(
        output_dir=arguments.output,
        seed=arguments.seed,
        fast=arguments.fast,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
