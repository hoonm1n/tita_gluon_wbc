"""VALIDATION_SIM: evaluate the 5k and 10k snapshots as each training job finishes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from diagnose_sim2sim_speed import REPO

ROS_PYTHON = Path.home() / "venvs/tita_wbc_jazzy/bin/python"
TEST = REPO / "test"
MID_CASES = ("forward_12", "reverse_12", "turn_left", "turn_right")


def wait_for_snapshot(out, label, step):
    while True:
        records = []
        for path in out.parent.glob("tita_mjlab_*_args.json"):
            args = json.loads(path.read_text())
            if args["run_name"] == f"speed_{label}":
                records.append(path)
        if len(records) > 1:
            raise ValueError(f"Ambiguous training runs for {label}")
        if records:
            run = records[0].name.removesuffix("_args.json")
            ckpt = REPO / "mj_lab/logs/flash_sac" / run / f"model_{step}"
            result_file = out.parent / f"{run}_result.json"
            if result_file.exists():
                result = json.loads(result_file.read_text())
                if result["evidence_status"] != "PASS":
                    raise RuntimeError(f"Training failure {label}: {result}")
            if (ckpt / "ddt.json").exists() and (step != 10000 or result_file.exists()):
                return ckpt
        time.sleep(10)


def evaluate(out, ckpt, midpoint):
    from sim2sim_ros2_jazzy.checkpoint import prepare_checkpoint, prepared_config

    cfg_path = prepared_config(ckpt) or prepare_checkpoint(ckpt, ROS_PYTHON)
    cfg = json.loads(cfg_path.read_text())
    manifest = dict(
        checkpoint=str(ckpt),
        sha256=cfg["policy_interface"]["checkpoint_sha256"],
        onnx_sha256=cfg["policy_interface"]["onnx_sha256"],
    )
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "complete.json"
    if marker.exists() and json.loads(marker.read_text()) == manifest:
        return
    with (out / "evaluation.log").open("w") as log:
        for variant in ("baseline", "impratio"):
            argv = [
                ROS_PYTHON,
                TEST / "diagnose_sim2sim_speed.py",
                out,
                "--workers",
                "8",
                "--variant",
                variant,
                "--finetune",
                "--checkpoint",
                ckpt,
            ]
            if midpoint:
                argv.extend(["--cases", *MID_CASES])
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
        if not midpoint:
            subprocess.run(
                [
                    sys.executable,
                    TEST / "diagnose_native_speed.py",
                    out,
                    "--finetune",
                    "--checkpoint",
                    ckpt,
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    marker.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    out = parser.parse_args().out.resolve()
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, str(REPO / "sim2sim/sim2sim_ros2_jazzy"))
    for seed in (42, 43, 44):
        for arm in "ABCD":
            label = f"{arm}_{seed}"
            for step in (5000, 10000):
                checkpoint = wait_for_snapshot(out, label, step)
                evaluate(out / label / str(step), checkpoint, step == 5000)
                print(f"[VALIDATION_SIM] {label} {step} evaluated", flush=True)
