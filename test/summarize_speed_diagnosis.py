"""VALIDATION_SIM and TRAINING_SIM: summarize speed diagnostics and plot paired turns."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnose_sim2sim_speed import conditions
from evaluate_sim2sim_heading import TRACKING_LIMITS, TRACKING_WINDOWS


def write_summary(out):
    rows = []
    for path in sorted(out.rglob("*.npz")):
        with np.load(path) as trace:
            if "condition" in trace:
                x, yaw, h, pitch = trace["condition"]
                data = trace["control"]
                assert data.shape == (3000, 15) and trace["physics"].shape == (12000, 11)
                # Check the actual scaled velocity/pose supplied to the policy.
                np.testing.assert_allclose(
                    trace["obs"][:, 6:9], np.tile([2 * x, 0, 0.25 * yaw], (3000, 1)), atol=1e-6
                )
                np.testing.assert_allclose(trace["obs"][-500:, 37], h, atol=1e-6)
                results = [(path.stem, x, yaw, h, pitch, data[:, 1:11])]
                runtime = "sim2sim"
                variant = "baseline" if path.parent == out else path.parent.name
            else:
                results = []
                runtime, variant = "native", path.stem
                for i, name in enumerate(trace["names"]):
                    with np.load(out / (name + ".npz")) as sim:
                        results.append((name, *sim["condition"], trace["values"][:, i]))
            for name, x, yaw, h, pitch, arr in results:
                for window, (start, end) in TRACKING_WINDOWS.items():
                    a = arr[round(start * 50) : round(end * 50)]
                    err = a[:, :3] - [x, 0, yaw]
                    mae = np.abs(err).mean(0)
                    row = dict(
                        runtime=runtime,
                        variant=variant,
                        case=name,
                        vx_cmd=x,
                        yaw_cmd=yaw,
                        height_cmd=h,
                        initial_pitch_delta_deg=pitch,
                        window=window,
                        samples=len(a),
                        vx=a[:, 0].mean(),
                        vy=a[:, 1].mean(),
                        yaw=a[:, 2].mean(),
                        vx_bias=err[:, 0].mean(),
                        vy_bias=err[:, 1].mean(),
                        yaw_bias=err[:, 2].mean(),
                        vx_mae=mae[0],
                        vy_mae=mae[1],
                        yaw_mae=mae[2],
                        speed_pass=int(all(mae < list(TRACKING_LIMITS.values()))),
                        height=a[:, 3].mean(),
                        pitch=a[:, 4].mean(),
                        wheel_target=a[:, 6:8].mean(),
                        wheel_actual=a[:, 8:10].mean(),
                    )
                    if runtime == "sim2sim":
                        sl = slice(round(start * 50), round(end * 50))
                        p = trace["physics"][round(start * 200) : round(end * 200)]
                        row.update(
                            axle_vx=data[sl, 11].mean(),
                            base_knee_contact_ticks=int(data[sl, 13:].any(axis=1).sum()),
                            wheel_saturated_frac=(np.abs(p[:, 1:3]) >= 12).mean(),
                            peak_torque=np.abs(p[:, 3:5]).max(),
                        )
                        for j, side in enumerate(("left", "right")):
                            touch = p[:, 9 + j] > 0
                            row.update(
                                {
                                    side + "_load_N": p[:, 5 + j].mean(),
                                    side + "_slip_m_s": p[touch, 7 + j].mean(),
                                }
                            )
                    rows.append(row)
    expected = {
        "baseline": len(conditions()),
        "impratio": len(conditions()),
        "integrator": 6,
        "armature": 6,
        "fresh_imu": 6,
        "native": 6,
        "native_impratio100": 6,
    }
    for variant, count in expected.items():
        assert sum(r["variant"] == variant and r["window"] == "late" for r in rows) == count, variant
    assert all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v, (float, np.floating)))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(out):
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex=True, sharey="row", layout="constrained")
    for col, (name, title) in enumerate(
        (("turn_left", "Left turn (+1 rad/s)"), ("turn_right", "Right turn (−1 rad/s)"))
    ):
        a = np.load(out / (name + ".npz"))["control"]
        t = a[:, 0]
        for ax, series, labels in zip(
            axes[:, col],
            [[a[:, 4]], [a[:, 5]], [a[:, 7:9].mean(1), a[:, 9:11].mean(1)], [a[:, 1], a[:, 11]]],
            [["Measured"], ["Measured"], ["Target", "Measured"], ["Base origin", "Axle midpoint"]],
        ):
            for y, label in zip(series, labels):
                ax.plot(t, y, label=label, linewidth=1.4)
            ax.grid(alpha=0.23)
        axes[0, col].axhline(0.335, color="0.4", linestyle=":", label="Target")
        axes[1, col].axhline(0, color="0.4", linestyle=":")
        axes[3, col].axhline(0, color="0.4", linestyle=":", label="Target")
        axes[0, col].set_title(title)
        axes[3, col].set_xlabel("Time (s)")
    for ax, label in zip(
        axes[:, 0], ["Height (m)", "Pitch (deg)", "Wheel mean (rad/s)", "Forward speed (m/s)"]
    ):
        ax.set_ylabel(label)
    for i in (0, 2, 3):
        axes[i, 1].legend(loc="best")
    fig.savefig(out / "turn_diagnosis.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    out = parser.parse_args().out
    write_summary(out)
    plot(out)
