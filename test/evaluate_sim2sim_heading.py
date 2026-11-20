"""VALIDATION_SIM: measure heading drift on fixed MuJoCo driving commands."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCENARIOS = (
    ("forward", (0.8, 0.0, 0.0), 20),
    ("reverse", (-0.8, 0.0, 0.0), 20),
    ("curve", (0.8, 0.0, 0.8), 6),
    ("turn_in_place", (0.0, 0.0, 1.0), 6),
)
POSE = (0.0, 0.0, 0.335)  # rad, rad, m
ALIGNMENT_SECONDS = 1.0
SPEED_TRANSIENT_SECONDS = 0.5
VELOCITY_FIELDS = ("vx_m_s", "vy_m_s", "yaw_rate_rad_s")
FIELDS = (
    "time_s", "world_x_m", "world_y_m", "world_yaw_rad", "longitudinal_m", "cross_track_m",
    "heading_change_deg", "roll_deg", "pitch_deg", "height_m",
    *(f"{prefix}_{axis}" for prefix in ("target", "actual") for axis in VELOCITY_FIELDS),
    "base_floor_contact", "knee_floor_contact",
)


def collect_sample(node, time_s, target):
    from sim2sim_ros2_jazzy.adapter import quat_wxyz_to_rotmat, quat_wxyz_to_rpy

    env = node.env
    quat = env.base_quat()
    roll, pitch, yaw = quat_wxyz_to_rpy(quat).astype(float)
    x, y, height = env.base_pos().astype(float)
    body_velocity = quat_wxyz_to_rotmat(quat).T @ env.base_lin_vel()
    actual = (float(body_velocity[0]), float(body_velocity[1]), float(env.base_ang_vel()[2]))
    if not np.isfinite([roll, pitch, yaw, x, y, height, *actual, *node.last_action]).all():
        raise RuntimeError(f"Non-finite state or policy output at {time_s:.3f} s")

    floor = env.model.geom("floor").id
    floor_bodies = set()
    for contact in env.data.contact:
        if contact.dist <= 0 and contact.efc_address >= 0 and floor in contact.geom:
            robot_geom = contact.geom[1] if contact.geom[0] == floor else contact.geom[0]
            floor_bodies.add(env.model.body(int(env.model.geom_bodyid[robot_geom])).name)

    sample = {
        "time_s": time_s, "world_x_m": x, "world_y_m": y, "world_yaw_rad": yaw,
        "roll_deg": float(np.degrees(roll)), "pitch_deg": float(np.degrees(pitch)), "height_m": height,
        "base_floor_contact": int("base_link" in floor_bodies),
        "knee_floor_contact": int(bool({"left_leg_3", "right_leg_3"} & floor_bodies)),
    }
    sample.update(zip((f"target_{axis}" for axis in VELOCITY_FIELDS), target))
    sample.update(zip((f"actual_{axis}" for axis in VELOCITY_FIELDS), actual))
    return sample


def run_scenario(config, command, duration, samples):
    import rclpy
    from geometry_msgs.msg import Twist, Vector3
    from sim2sim_ros2_jazzy.sim2sim_node import Sim2SimNode

    rclpy.init(args=["--ros-args", "-p", f"config:={config}", "-p", "viewer:=false"])
    node = None
    try:
        node = Sim2SimNode()
        dt = node.env.control_dt
        node._cmd_pose_cb(Vector3(x=POSE[0], y=POSE[1], z=POSE[2]))
        for _ in range(round(5.0 / dt)):
            node._step()
            if node.sm.state == "rl":
                break
        else:
            raise RuntimeError("Sim2sim did not enter RL within 5 s of preparation")

        for _ in range(round(ALIGNMENT_SECONDS / dt)):
            node._step()
            if node.sm.state != "rl":
                raise RuntimeError("Sim2sim left RL before the driving command")
        samples.append(collect_sample(node, 0.0, command))

        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = command
        node._cmd_twist_cb(message)
        if not np.allclose(node.command, command):
            raise RuntimeError(f"Requested command {command} was not applied: {node.command}")
        for step in range(1, round(duration / dt) + 1):
            node._step()
            if node.sm.state != "rl" or not node.env.is_running():
                raise RuntimeError(f"Sim2sim left RL at {step * dt:.3f} s")
            samples.append(collect_sample(node, step * dt, node.command.astype(float).tolist()))
    finally:
        try:
            if node is not None:
                node.env.close()
                node.destroy_node()
        finally:
            rclpy.try_shutdown()


def add_relative_measurements(samples):
    if not samples:
        return
    start = samples[0]
    yaw = np.unwrap([sample["world_yaw_rad"] for sample in samples])
    heading = np.degrees(yaw - yaw[0])
    cosine, sine = np.cos(yaw[0]), np.sin(yaw[0])
    for sample, angle in zip(samples, heading):
        dx = sample["world_x_m"] - start["world_x_m"]
        dy = sample["world_y_m"] - start["world_y_m"]
        sample.update(longitudinal_m=float(cosine * dx + sine * dy),
                      cross_track_m=float(-sine * dx + cosine * dy), heading_change_deg=float(angle))


def window_metrics(samples):
    if not samples:
        return None
    result = {"measured_seconds": samples[-1]["time_s"]}
    for field in ("longitudinal_m", "cross_track_m", "heading_change_deg"):
        result[f"final_{field}"] = samples[-1][field]
        result[f"max_abs_{field}"] = max(abs(row[field]) for row in samples)
    measured = [row for row in samples if row["time_s"] > SPEED_TRANSIENT_SECONDS]
    if measured:
        result["speed_mae"] = {
            axis: float(np.mean([abs(row[f"actual_{axis}"] - row[f"target_{axis}"]) for row in measured]))
            for axis in VELOCITY_FIELDS
        }
        pose = zip(("roll_deg", "pitch_deg", "height_m"), (0.0, 0.0, POSE[2]))
        result["pose_mae"] = {field: float(np.mean([abs(row[field] - goal) for row in measured]))
                              for field, goal in pose}
    for field in ("base_floor_contact", "knee_floor_contact"):
        result[f"{field}_ticks"] = sum(row[field] for row in samples if row["time_s"] > 0)
    return result


def measure(checkpoint, output):
    sys.path.insert(0, str(REPO / "sim2sim/sim2sim_ros2_jazzy"))
    from sim2sim_ros2_jazzy.checkpoint import prepared_config

    config = prepared_config(checkpoint)
    if config is None:
        raise RuntimeError("Use the sim2sim launcher with --prepare-only to verify the checkpoint first")
    cfg = json.loads(config.read_text())
    interface = cfg["policy_interface"]
    output.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for name, command, duration in SCENARIOS:
        samples = []
        status, detail = "PASS", "Completed the requested command window"
        try:
            run_scenario(config, command, duration, samples)
        except Exception as error:
            status, detail = "FAIL", f"{type(error).__name__}: {error}"
        add_relative_measurements(samples)
        csv_path = output / f"{name}.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(samples)
        metrics = window_metrics(samples)
        if status == "PASS" and (metrics["base_floor_contact_ticks"] or metrics["knee_floor_contact_ticks"]):
            status, detail = "SUSPICIOUS", "Completed with base or knee contact on the floor"
        windows = {str(duration): metrics}
        if duration > 6:
            windows["6"] = window_metrics([row for row in samples if row["time_s"] <= 6])
        scenarios.append({"name": name, "command": command, "requested_seconds": duration, "csv": csv_path.name,
                          "evidence_status": status, "detail": detail, "windows_s": windows})
        print(f"[VALIDATION_SIM] {name} {status} - {detail}", flush=True)

    statuses = {scenario["evidence_status"] for scenario in scenarios}
    result = {
        "execution_domain": "VALIDATION_SIM",
        "evidence_status": next(status for status in ("FAIL", "SUSPICIOUS", "PASS") if status in statuses),
        "checkpoint": interface["checkpoint"], "checkpoint_sha256": interface["checkpoint_sha256"],
        "onnx_sha256": interface["onnx_sha256"], "model": cfg["model"], "control": cfg["control"],
        "pose_target_rad_m": POSE, "zero_command_rl_seconds": ALIGNMENT_SECONDS,
        "speed_mae_excluded_seconds": SPEED_TRANSIENT_SECONDS, "scenarios": scenarios,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result["evidence_status"] != "FAIL"


def plot(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads((output / "result.json").read_text())
    figure, (track, heading) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    figure.suptitle(f"Sim2sim driving  |  {Path(result['checkpoint']).parent.name}")
    for scenario in result["scenarios"]:
        samples = np.genfromtxt(output / scenario["csv"], delimiter=",", names=True, ndmin=1)
        if not len(samples):
            continue
        label = scenario["name"].replace("_", " ").title()
        line, = track.plot(samples["longitudinal_m"], samples["cross_track_m"], label=label)
        track.scatter(samples["longitudinal_m"][-1], samples["cross_track_m"][-1], color=line.get_color(), s=20)
        if scenario["name"] in ("forward", "reverse"):
            heading.plot(samples["time_s"], samples["heading_change_deg"], label=label, color=line.get_color())
    track.set(title="Track relative to the starting heading", xlabel="Longitudinal distance (m)", ylabel="Lateral distance (m)")
    track.set_aspect("equal", adjustable="datalim")
    heading.set(title="Heading change during straight commands", xlabel="Time after command (s)", ylabel="Heading change (deg)")
    heading.axvline(6.0, color="0.55", linestyle="--", linewidth=1)
    for axes in (track, heading):
        axes.axhline(0.0, color="0.6", linewidth=0.8)
        axes.grid(alpha=0.25)
        if axes.get_legend_handles_labels()[0]:
            axes.legend()
    figure.savefig(output / "heading_trajectory.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Experiment directory for CSV, JSON and figure")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path, help="Prepared sim2sim checkpoint to measure")
    mode.add_argument("--plot-only", action="store_true", help="Render the saved heading and trajectory figure")
    arguments = parser.parse_args()
    if arguments.plot_only:
        plot(arguments.output)
    else:
        sys.exit(0 if measure(arguments.checkpoint, arguments.output) else 1)
