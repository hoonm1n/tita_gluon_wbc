"""VALIDATION_SIM: score the paired speed fine-tuning experiment with fixed acceptance tests."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnose_sim2sim_speed import CKPT, POSE_CASES, conditions, native_tuning_names, tuning_cases
from evaluate_sim2sim_heading import TRACKING_WINDOWS

SIM = ("sim100", "sim1")
DOMAINS = (*SIM, "native")
CASES = {name: case for case, name in tuning_cases()}
SCREEN_CASES = set(conditions().values())
MID = {"forward_12", "reverse_12", "turn_left", "turn_right"}


def evaluate_rollout(label, domain, name, data, contacts, axle):
    x, yaw, height, initial_pitch = CASES[name]
    roll_goal, pitch_goal = POSE_CASES.get(name, (0.0, 0.0))
    assert data.shape == (3000, 10) and np.isfinite(data).all() and np.isfinite(axle).all()
    rows = []
    for window, (start, end) in TRACKING_WINDOWS.items():
        sl = slice(round(start * 50), round(end * 50))
        a = data[sl]
        error = a[:, :3] - [x, 0.0, yaw]
        mae = np.abs(error).mean(axis=0)
        rows.append(
            dict(
                label=label,
                domain=domain,
                case=name,
                window=window,
                samples=len(a),
                x=x,
                yaw=yaw,
                height_goal=height,
                start_pitch=initial_pitch,
                vx=a[:, 0].mean(),
                vy=a[:, 1].mean(),
                wz=a[:, 2].mean(),
                vx_mae=mae[0],
                vy_mae=mae[1],
                wz_mae=mae[2],
                vx_bias=error[:, 0].mean(),
                vy_bias=error[:, 1].mean(),
                wz_bias=error[:, 2].mean(),
                height_mae=np.abs(a[:, 3] - height).mean(),
                pitch_mae=np.abs(a[:, 4] - pitch_goal).mean(),
                roll_mae=np.abs(a[:, 5] - roll_goal).mean(),
                wheel_target=a[:, 6:8].mean(),
                wheel_actual=a[:, 8:10].mean(),
                axle_vx=axle[sl].mean(),
                contacts_60s=int(contacts.any(axis=1).sum()),
                speed_pass=bool(all(mae < [0.10, 0.08, 0.10])),
            )
        )
    return rows


def score(label, folder, midpoint=False):
    rows = []
    for domain, subdir in zip(SIM, (folder, folder / "impratio")):
        for name in sorted(MID if midpoint else CASES):
            with np.load(subdir / (name + ".npz")) as z:
                a = z["control"]
                np.testing.assert_allclose(z["condition"], CASES[name])
                goal = np.array([2 * CASES[name][0], 0, 0.25 * CASES[name][1]])
                assert np.allclose(z["obs"][:, 6:9], goal, atol=1e-6)
                pose = [*np.deg2rad(POSE_CASES.get(name, (0.0, 0.0))), CASES[name][2]]
                assert np.allclose(z["obs"][-500:, 35:38], pose, atol=1e-6)
                rows.extend(evaluate_rollout(label, domain, name, a[:, 1:11], a[:, 13:], a[:, 11]))
    if not midpoint:
        with np.load(folder / "native.npz") as z:
            assert set(z["names"]) == set(native_tuning_names())
            vals, contacts, axle, obs = z["values"], z["contacts"], z["axle"], z["obs"]
            for i, name in enumerate(z["names"]):
                pose = [*np.deg2rad(POSE_CASES.get(name, (0.0, 0.0))), CASES[name][2]]
                assert np.allclose(obs[-500:, i, 35:38], pose, atol=1e-6)
                rows.extend(
                    evaluate_rollout(label, "native", name, vals[:, i], contacts[:, i], axle[:, i, 0])
                )
    return rows


def screen(rows, domain):
    return {
        name
        for name in SCREEN_CASES
        if all(
            r["speed_pass"]
            for r in rows
            if r["case"] == name and r["domain"] == domain and r["window"] in ("overall", "late")
        )
    }


def judge(rows, source):
    baseline = {(r["domain"], r["case"], r["window"]): r for r in source}
    summary = {"label": rows[0]["label"]}
    for d in SIM:
        passed = screen(rows, d)
        summary[d + "_pass"] = len(passed)
        summary[d + "_lost"] = len(screen(source, d) - passed)
    hi12, hi14, turn_v, turn_w, pose_a, pose_h = [], [], [], [], [], []
    for r in rows:
        name, (x, yaw, h, p) = r["case"], CASES[r["case"]]
        if name in POSE_CASES:
            if r["window"] in ("late", "overall"):
                ref = baseline[r["domain"], name, r["window"]]
                axis = "roll_mae" if name.startswith("roll") else "pitch_mae"
                pose_a.append(r[axis] - ref[axis])
                pose_h.append(r["height_mae"] - ref["height_mae"])
        elif p == 0 and h == 0.335 and r["window"] == "late":
            if x in (1.2, 1.4) and yaw == 0:
                (hi12 if x == 1.2 else hi14).append(r["vx_mae"])
            if x == 0 and abs(yaw) in (0.5, 1.0, 1.5):
                turn_v.append(r["vx_mae"])
                turn_w.append(r["wz_mae"])
    assert len(hi12) == len(hi14) == 3 and len(turn_v) == 18
    summary.update(
        vx12=max(hi12),
        vx14=max(hi14),
        turn_vx=max(turn_v),
        turn_wz=max(turn_w),
        pose_deg=max(pose_a),
        pose_m=max(pose_h),
        contact_ticks=sum(r["contacts_60s"] for r in rows if r["window"] == "late"),
    )
    # Tie-break compares the two sim2sim contact settings, after every gate has passed.
    ratios = []
    for r in rows:
        x, y, h, p = CASES[r["case"]]
        if r["domain"] not in SIM or r["window"] != "late" or p or h != 0.335 or r["case"] in POSE_CASES:
            continue
        if not y and x in (1.2, 1.4):
            ratios.append(r["vx_mae"] / (0.05 if x == 1.2 else 0.08))
        if not x and abs(y) in (0.5, 1.0, 1.5):
            ratios.extend((r["vx_mae"] / 0.05, r["wz_mae"] / 0.1))
    summary["worst_ratio"] = max(ratios)
    summary["accepted"] = bool(
        max(hi12) <= 0.05
        and max(hi14) <= 0.08
        and max(turn_v) <= 0.05
        and max(turn_w) < 0.10
        and summary["sim100_lost"] == summary["sim1_lost"] == summary["contact_ticks"] == 0
        and max(pose_a) <= 1
        and max(pose_h) <= 0.01
    )
    return summary


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    shapes = [(0, 0, "vx", "Forward response (m/s)"), (1, 0, "vx", "Turn residual (m/s)")]
    for col, d in enumerate(DOMAINS):
        for row, _, key, title in shapes:
            ax = axes[row, col]
            for label in ("source", *"ABCD"):
                matching = [
                    r
                    for r in rows
                    if r["domain"] == d
                    and r["window"] == "late"
                    and (
                        r["label"] == "source"
                        if label == "source"
                        else r["label"].startswith(label + "_")
                    )
                    and r["height_goal"] == 0.335
                    and r["start_pitch"] == 0
                    and r["case"] not in POSE_CASES
                ]
                matching = [
                    r
                    for r in matching
                    if (r["yaw"] == 0 and r["x"] > 0 if row == 0 else r["x"] == 0 and r["yaw"] != 0)
                ]
                cmd = "x" if row == 0 else "yaw"
                xs = sorted({r[cmd] for r in matching})
                ax.plot(
                    xs, [median(r[key] for r in matching if r[cmd] == x) for x in xs], ".-", label=label
                )
            if row == 0:
                ax.plot(xs, xs, ":", color="0.5")
            else:
                ax.axhline(0, color="0.5", linestyle=":")
            ax.set(xlabel="Command (m/s)" if row == 0 else "Command (rad/s)", ylabel=title, title=d)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out / "speed_response.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), layout="constrained")
    for col, d in enumerate(DOMAINS):
        for i, name in enumerate(("turn_left", "turn_right")):
            a = [r for r in rows if r["domain"] == d and r["case"] == name and r["window"] == "late"]
            ax = axes[i, col]
            labels = ["source", *"ABCD"]
            for field, style in (("wheel_target", "o-"), ("wheel_actual", "s--")):
                ax.plot(
                    labels,
                    [
                        median(r[field] for r in a if r["label"].split("_")[0] == group)
                        for group in labels
                    ],
                    style,
                    label=field,
                )
            ax.set(title=f"{d} · {name}", ylabel="Wheel mean (rad/s)")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out / "turn_wheels.png", dpi=150)
    plt.close(fig)


def main(out, source_only):
    source = score("source", out / "source")
    assert [len(screen(source, d)) for d in SIM] == [49, 55]
    write_csv(out / "source_windows.csv", source)
    if source_only:
        print("PASS source 49/58, 55/58; native and pose evaluation complete")
        return
    all_rows = list(source)
    interim, summary = [], []
    for seed in (42, 43, 44):
        for arm in "ABCD":
            label = f"{arm}_{seed}"
            folder = out / label
            rows = score(label, folder / "10000")
            all_rows.extend(rows)
            interim.extend(score(label, folder / "5000", midpoint=True))
            entry = judge(rows, source)
            entry.update(json.loads((folder / "10000/complete.json").read_text()))
            summary.append(entry)
    write_csv(out / "final_windows.csv", all_rows)
    write_csv(out / "midpoint_windows.csv", interim)
    write_csv(out / "seed_results.csv", summary)
    eligible = []
    metrics = ("vx12", "vx14", "turn_vx", "turn_wz")
    reference = judge(source, source)
    pose_failures = sum(r["pose_deg"] > 1 or r["pose_m"] > 0.01 for r in summary)
    contact_rollouts = sum(r["contacts_60s"] > 0 for r in all_rows if r["window"] == "late")
    lines = [
        "# 속도 추종 미세학습 2×2 비교",
        "",
        "실행 도메인 `TRAINING_SIM` · `VALIDATION_SIM`",
        "",
        "4096개 환경에서 각 10,000스텝을 학습했다. 세 seed 중 두 개 이상이 모든 사전 기준을 통과해야 채택한다.",
        "",
        "| 군 | 통과 seed | +1.2 오차 | +1.4 오차 | 회전 vx 오차 | 회전 yaw 오차 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| 원본 | — | " + " | ".join(f"{reference[key]:.4f}" for key in metrics) + " |",
    ]
    for arm in "ABCD":
        group = [r for r in summary if r["label"].startswith(arm + "_")]
        passed = [r for r in group if r["accepted"]]
        if len(passed) >= 2:
            eligible.append((median(r["worst_ratio"] for r in group), arm))
        lines.append(
            f"| {arm} | {len(passed)}/3 | "
            + " | ".join(f"{median(r[key] for r in group):.4f}" for key in metrics)
            + " |"
        )
    indexed = {r["label"]: r for r in summary}
    effects = []
    for seed in (42, 43, 44):
        for arm, control in (("B", "A"), ("C", "A"), ("D", "B"), ("D", "C")):
            candidate, ref = indexed[f"{arm}_{seed}"], indexed[f"{control}_{seed}"]
            effects.append(
                dict(
                    seed=seed,
                    contrast=f"{arm}-{control}",
                    **{
                        key: candidate[key] - ref[key]
                        for key in (
                            "vx12",
                            "vx14",
                            "turn_vx",
                            "turn_wz",
                            "sim100_lost",
                            "sim1_lost",
                            "pose_deg",
                            "pose_m",
                        )
                    },
                )
            )
    write_csv(out / "paired_effects.csv", effects)
    selected = min(eligible)[1] if eligible else "model_85000"
    if eligible:
        best = min(
            (r for r in summary if r["accepted"] and r["label"].startswith(selected + "_")),
            key=lambda r: r["worst_ratio"],
        )
        selected = best["checkpoint"]
    lines += [
        "",
        f"**채택 결과** `{selected}`",
        "",
        "각 열은 seed별 최악 조건 MAE의 중앙값이다. 선속도는 m/s, yaw rate는 rad/s다.",
        f"자세·높이 회귀 기준 초과는 최종 후보 {pose_failures}/{len(summary)}개였다.",
        f"원본 포함 최종 60초 시험 {len(all_rows) // len(TRACKING_WINDOWS)}회 중 "
        f"차체·무릎 접촉이 관측된 시험은 {contact_rollouts}회였다. 접촉은 제어 주기마다 집계했다.",
        "원본 actor SHA256 `" + json.loads((CKPT / "ddt.json").read_text())["actor_sha256"] + "`",
        "",
        "[seed별 판정과 해시](seed_results.csv) · [최종 구간 지표](final_windows.csv) · [5,000스텝 진단](midpoint_windows.csv)",
        "",
        "[속도 응답](speed_response.png) · [좌우 회전 바퀴 목표·실측](turn_wheels.png)",
        "",
        "[같은 seed의 실험군 간 차이](paired_effects.csv) · [시험 조건 및 판정 기준](protocol.md)",
        "",
        "실행 상태 PASS. 후보 채택 판정은 위 표의 통과 seed 수를 사용한다.",
    ]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    plot(all_rows, out)
    print("Selected", selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    main(args.out, args.source_only)
