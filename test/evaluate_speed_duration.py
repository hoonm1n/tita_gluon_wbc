"""VALIDATION_SIM: evaluate duration-ablation snapshots and compare with saved B runs."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from report_speed_finetune import DOMAINS, TRACKING_WINDOWS, judge, score, write_csv
from run_speed_finetune_eval import REPO, evaluate

SEEDS = (42, 43, 44)
ARMS = ("E", "H")
PREVIOUS = REPO / "docs/exp/2026-11-22/speed_finetune"
METRICS = ("vx12", "vx14", "turn_vx", "turn_wz", "pose_deg", "pose_m")


def training_record(out, label):
    run_name = f"speed_duration_{label}"
    matches = [
        path.name.removesuffix("_args.json")
        for path in out.parent.glob("tita_mjlab_*_args.json")
        if json.loads(path.read_text())["run_name"] == run_name
    ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous training records for {label}")
    if not matches:
        return None
    run = matches[0]
    path = out.parent / f"{run}_result.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (record["evidence_status"], record["step"], record["updates"]) != ("PASS", 10000, 19948):
        raise ValueError(f"Training failed or did not match the prior budget for {label}: {record}")
    return run, record


def wait_for_training(out, label):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        record = training_record(out, label)
        if record:
            return record
        time.sleep(10)
    raise TimeoutError(f"Missing completed training record for {label}")


def analyze(out):
    source = score("source", PREVIOUS / "source")
    windows = list(source)
    scores, exposure, turns = [], [], []
    for seed in SEEDS:
        for arm in ("B", *ARMS):
            label = f"{arm}_{seed}"
            folder = PREVIOUS / label / "10000" if arm == "B" else out / label
            rows = score(label, folder)
            assert len(rows) == len(TRACKING_WINDOWS) * (62 + 62 + 16)
            windows.extend(rows)
            result = judge(rows, source)
            manifest = json.loads((folder / "complete.json").read_text())
            result.update(manifest)
            scores.append(result)
            if arm in ARMS:
                run, record = training_record(out, label)
                assert record["checkpoint_sha256"] == manifest["sha256"]
                metrics = out.parent / f"{run}_metrics.jsonl"
                ticks = json.loads(metrics.read_text().splitlines()[-1])
                assert ticks["step"] == 10000
                exposure.append(dict(
                    label=label,
                    turn_fraction=ticks["env/pure_turn_step_fraction"],
                    long_turn_fraction=ticks["env/pure_turn_at_least_15s_step_fraction"],
                    args_log=f"../{run}_args.json",
                    metrics_log=f"../{metrics.name}",
                    training_result=f"../{run}_result.json",
                ))
            for domain in DOMAINS:
                for window in ("early", "late"):
                    pure_turns = [
                        r for r in rows
                        if r["domain"] == domain and r["window"] == window and r["x"] == 0
                        and abs(r["yaw"]) in (0.5, 1.0, 1.5) and r["height_goal"] == 0.335
                        and r["start_pitch"] == 0
                    ]
                    assert len(pure_turns) == 6
                    turns.append(dict(
                        label=label, domain=domain, window=window,
                        **{key: max(r[key] for r in pure_turns) for key in
                           ("vx_mae", "vy_mae", "wz_mae", "pitch_mae", "height_mae")},
                    ))

    indexed = {r["label"]: r for r in scores}
    paired = [
        dict(seed=seed, contrast=f"{arm}-{base}", **{
            key: indexed[f"{arm}_{seed}"][key] - indexed[f"{base}_{seed}"][key]
            for key in (*METRICS, "sim100_lost", "sim1_lost")
        })
        for seed in SEEDS for arm, base in (("E", "B"), ("H", "E"))
    ]
    for name, rows in (("final_windows", windows), ("seed_results", scores),
                       ("training_exposure", exposure), ("paired_effects", paired), ("turn_windows", turns)):
        write_csv(out / f"{name}.csv", rows)
    print(f"[VALIDATION_SIM] {len(scores)} trained policies and the source, "
          f"{len(windows) // len(TRACKING_WINDOWS)} total 60 s runs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, str(REPO / "sim2sim/sim2sim_ros2_jazzy"))
    if not args.analyze_only:
        for seed in SEEDS:
            for arm in ARMS:
                label = f"{arm}_{seed}"
                _, record = wait_for_training(out, label)
                evaluate(out / label, Path(record["checkpoint"]), midpoint=False)
                print(f"[VALIDATION_SIM] {label} evaluated", flush=True)
    analyze(out)


if __name__ == "__main__":
    main()
