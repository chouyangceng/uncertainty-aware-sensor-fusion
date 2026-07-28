from pathlib import Path


def test_sensor_fault_experiment_writes_summary():
    from uncertainty_sensor_fusion.experiments import run

    summary = run(seed=3, frames=30, output_dir=Path("artifacts/test-fusion"))
    assert summary["baseline"]
    assert (Path("artifacts/test-fusion") / "summary.json").exists()
