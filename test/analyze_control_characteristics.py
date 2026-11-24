"""DESKTOP_DRY_RUN: aggregate TITA traces without stepping either simulator."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_control_characteristics import OLD, REPO

CHANNELS = ("vx", "vy", "wz", "roll", "pitch", "height")
UNITS = ("m/s", "m/s", "rad/s", "rad", "rad", "m")
GT = (0, 1, 5, 6, 7, 9)
WHEELS = (3, 7)
LEGS = (0, 1, 2, 4, 5, 6)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def tracking(error):
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))),
                bias=float(error.mean()), error_std=float(error.std()))


def longest_episode(mask, dt):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return float(np.max(np.diff(edges)[::2], initial=0) * dt)


def step_metrics(t, y, initial, target, pre_std, dt):
    delta = target - initial
    if abs(delta) < 1e-9:
        return {}
    fraction = (y - initial) / delta
    first = [np.flatnonzero(fraction >= bound) for bound in (.1, .9)]
    tolerance = max(.05 * abs(delta), 3 * pre_std)
    settled = np.abs(y - target) <= tolerance
    start = np.flatnonzero(np.diff(np.r_[False, settled, False].astype(int)) == 1)
    end = np.flatnonzero(np.diff(np.r_[False, settled, False].astype(int)) == -1)
    eligible = start[(end-start)*dt >= 1 - 1e-8]
    settle_idx = eligible[0] if eligible.size else None
    return dict(rise_s=float(t[first[1][0]]-t[first[0][0]]) if first[1].size else None,
                settle_s=float(t[settle_idx]) if settle_idx is not None else None,
                observation_end_s=float(t[-1]), settling_band=float(tolerance),
                reexit=bool(not settled[settle_idx:].all()) if settle_idx is not None else None,
                overshoot_pct=float(100*max(fraction.max()-1, 0)))


def archived_trajectory():
    rows = []
    for name in ("forward_04", "forward_08", "forward_12", "reverse_04", "reverse_08", "reverse_12"):
        path = OLD / "sim2sim_speed_tracking_60s" / (name + ".csv")
        z = np.genfromtxt(path, names=True, delimiter=",")
        for duration in (6, 20, 60):
            sample = z[z["time_s"] <= duration]
            time = sample["time_s"]
            yaw = sample["heading_change_deg"]
            cross = sample["cross_track_m"]
            roll, pitch, theta = (np.deg2rad(sample[k]) for k in ("roll_deg", "pitch_deg", "heading_change_deg"))
            # Rotate measured planar velocity into the starting-heading plane.
            along = sample["actual_vx_m_s"] * np.sin(theta) * np.cos(pitch)
            lateral = sample["actual_vy_m_s"] * (np.sin(theta)*np.sin(pitch)*np.sin(roll)
                                                 + np.cos(theta)*np.cos(roll))
            integrate = getattr(np, "trapezoid", np.trapz if hasattr(np, "trapz") else None)
            heading_m, lateral_m = integrate(along, time), integrate(lateral, time)
            rows.append(dict(case=name, seconds=duration, heading_rate_deg_s=np.polyfit(time, yaw, 1)[0],
                             heading_final_deg=yaw[-1], heading_max_deg=np.abs(yaw).max(),
                             cross_final_m=cross[-1], cross_max_m=np.abs(cross).max(),
                             cross_from_vx_m=heading_m, cross_from_vy_m=lateral_m,
                             cross_unexplained_m=cross[-1]-heading_m-lateral_m,
                             cross_per_abs_m=cross[-1]/max(abs(sample["longitudinal_m"][-1]), 1e-9),
                             source=str(path.relative_to(OLD.parent.parent.parent))))
    return rows


def fits(time, values, frequency):
    angle = 2 * np.pi * frequency * time
    design = np.c_[np.ones_like(angle), time-time.mean(), np.sin(angle), np.cos(angle)]
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    amplitude = np.hypot(coefficients[2], coefficients[3])
    phase = np.arctan2(coefficients[3], coefficients[2])
    rem = values - design @ coefficients
    return amplitude, phase, np.sqrt(np.mean(rem**2))


def wheel_odometry(sim):
    """No-slip base vx from wheel contact Jacobians at the same pre-step state."""
    contacts = sim["contacts"]
    q = sim["qpos_pre"]
    v = sim["qvel_pre"]
    # World projection of base x, with the full quaternion rotation.
    w, x, y, z = q[:, 3:7].T
    bx = np.c_[1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)]
    n = contacts[:, :, 8:11]
    t = bx[:, None]-n*np.sum(n*bx[:, None], axis=-1, keepdims=True)
    t /= np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12)
    gt = np.einsum("ti,tki->tk", v[:, :3], t)
    physical = gt - contacts[:, :, 2]  # v_contact = v_base + J_relative(q) dq
    both = (contacts[:, :, 0] > 0).all(axis=1)
    simple = .0925*v[:, np.asarray(WHEELS)+6].mean(axis=1)
    return np.c_[gt.mean(axis=1), simple, physical.mean(axis=1)], both


def load_traces(run):
    with np.load(run / "policy.npz") as z:
        p = dict(z)
    with np.load(run / "physics.npz") as z:
        f = dict(z)
    manifest = json.loads((run / "manifest.json").read_text())
    for signal in (p, f):
        if not all(np.isfinite(value).all() for value in signal.values()):
            raise ValueError(f"Nonfinite telemetry in {run}")
    return manifest, p, f


def characterize(run):
    meta, p, f = load_traces(run)
    dt, dp = meta["dt_policy"], meta["dt_physics"]
    rows, joint_rows, phase_rows, frf_rows = [], [], [], []
    body = p["body_gt"][:, GT]
    applied = p["command_applied"]
    qvel = f["qvel_pre"][:, 6:]
    qpos = f["qpos_pre"][:, 7:]
    q_error = f["target"][:, LEGS] - qpos[:, LEGS]
    dq_error = f["target"][:, WHEELS] - qvel[:, WHEELS]
    joint_error = np.empty_like(qvel)
    joint_error[:, LEGS] = q_error
    joint_error[:, WHEELS] = dq_error
    command_norm = p["normalized_action"]
    y_kinematic, both = wheel_odometry(f)
    with (run / "events.csv").open() as stream:
        events = list(csv.DictReader(stream))
    for phase in np.unique(p["phase"]):
        phase_select = p["phase"] == phase
        dt_seg, speed, pose, ramp = meta["segments"][phase]
        t = p["phase_time"][phase_select] + dt
        phase_idx = np.flatnonzero(phase_select)[0]
        prior = body[max(0, phase_idx-round(1/dt)):phase_idx]
        if not prior.size:
            init = np.load(run / "initial.npz")
            initial = infer_initial(init["qpos"], init["qvel"])
            pre_std = np.zeros(6)
        else:
            initial = prior[-1]
            pre_std = prior.std(axis=0)
        y, cmd = body[phase_select], applied[phase_select]
        nominal = np.array([speed[0], 0, speed[1], *pose])
        phase_info = dict(run=run.name, phase=phase, duration_s=t[-1],
                          complete=t[-1] >= dt_seg - dt / 2, event_count=len(events))
        world_xy = p["qpos"][phase_select, :2]
        heading = np.unwrap(p["body_gt"][phase_select, 8])
        crossing = np.flatnonzero(np.sign(y[:,0]) != np.sign(initial[0])) if phase_idx else np.empty(0, int)
        latches = np.flatnonzero(p["position_latched"][phase_select])
        first_cross = crossing[0] if crossing.size else None
        xy_origin = p["qpos"][phase_idx-1,:2] if phase_idx else np.load(run/"initial.npz")["qpos"][:2]
        phase_rows.append(dict(**phase_info, displacement_m=np.linalg.norm(world_xy[-1]-xy_origin),
                               vx_zero_cross_s=t[first_cross] if first_cross is not None else None,
                               vx_zero_cross_displacement_m=np.linalg.norm(world_xy[first_cross]-xy_origin)
                               if first_cross is not None else None,
                               first_stand_latch_s=t[latches[0]] if latches.size else None,
                               heading_delta_deg=np.degrees(heading[-1]-heading[0]),
                               abs_heading_peak_deg=np.degrees(np.max(np.abs(heading-heading[0]))),
                               first_vx=y[0, 0], last_vx=y[-1, 0]))
        variant = meta["variant"]
        if variant.startswith("sine_"):
            name, freq = variant.split("=")
            freq = float(freq)
            axis = 0 if name == "sine_vx" else 2
            select = t > 2 / freq
            # Match output at post-step timestamps to the held input stream.
            gain_u, phase_u, _ = fits(t[select]-dt, cmd[select, axis], freq)
            gain_y, phase_y, residual = fits(t[select], y[select, axis], freq)
            phase_rad = (phase_y-phase_u+np.pi) % (2*np.pi)-np.pi
            reference = f["t_pre"] > 2/freq
            wheel_mix = np.array([.5, .5] if axis == 0 else [-.5, .5])
            wheel_target = f["target"][reference][:, WHEELS] @ wheel_mix
            wheel_actual = qvel[reference][:, WHEELS] @ wheel_mix
            target_amplitude, target_phase, _ = fits(f["t_pre"][reference], wheel_target, freq)
            actual_amplitude, actual_phase, _ = fits(f["t_pre"][reference], wheel_actual, freq)
            wheel_phase = (actual_phase-target_phase+np.pi) % (2*np.pi)-np.pi
            frf_rows.append(dict(run=run.name, signal=CHANNELS[axis], hz=freq,
                                 gain=gain_y/gain_u, phase_deg=np.degrees(phase_rad),
                                 apparent_delay_ms=-1e3*phase_rad/(2*np.pi*freq),
                                 fundamental_amplitude=gain_y, fit_residual_rms=residual,
                                 wheel_target_gain=actual_amplitude/target_amplitude,
                                 wheel_target_phase_deg=np.degrees(wheel_phase)))
        for label, begin, end in (("initial", 0, .5), ("transition", .5, 2),
                                  ("medium", 2, 10), ("early", 10, 20), ("legacy_overall", .5, 60),
                                  ("final10", max(0, dt_seg-10), dt_seg)):
            mask = (t > begin + 1e-9) & (t <= end + 1e-9)
            if not mask.any():
                continue
            for j, (channel, unit) in enumerate(zip(CHANNELS, UNITS)):
                error = (y[:, j] - cmd[:, j])[mask]
                row = dict(**phase_info, case=meta["case"], variant=variant, seed=meta["seed"],
                           residual=meta["residual"], window=label, channel=channel, unit=unit,
                           n=int(mask.sum()), command_last=cmd[mask, j][-1],
                           mean=float(y[mask, j].mean()), **tracking(error),
                           peak_deviation=float(np.abs(y[mask, j]-initial[j]).max()),
                           error_slope=float(np.polyfit(t[mask], error, 1)[0]) if mask.sum() > 1 else None)
                prior_command = applied[phase_idx-1, j] if phase_idx else (0.335 if j == 5 else 0.)
                if label == "initial" and not np.isclose(nominal[j], prior_command):
                    row.update(step_metrics(t, y[:, j], initial[j], nominal[j], pre_std[j], dt))
                    command_response = step_metrics(t-dt, cmd[:, j], initial[j], nominal[j], 0., dt)
                    row["command_rise_s"] = command_response.get("rise_s")
                rows.append(row)
        pmask = phase_select[f["policy_tick"]]
        for scope, select in (("whole_phase", pmask),
                               ("final10", pmask & (f["t_post"] > p["t_post"][phase_select][0] - dt + dt_seg-10 + 1e-8))):
            if not select.any():
                continue
            for j, name in enumerate(meta["joints"]):
                tau = f["qfrc_actuator"][select, j]
                sat = np.abs(f["preclip"][select, j]-f["postclip"][select, j]) > 1e-5
                clip = np.abs(f["target_raw"][select, j]-f["target"][select, j]) > 1e-5
                joint_rows.append(dict(run=run.name, phase=phase, window=scope, joint=name,
                                       tracking_unit="rad/s" if j in WHEELS else "rad",
                                       **tracking(joint_error[select, j]),
                                       action_near_limit_frac=np.mean(np.abs(command_norm[phase_select, j]) >= .98),
                                       target_clip_frac=clip.mean(), target_clip_longest_s=longest_episode(clip, dp),
                                       torque_clip_frac=sat.mean(), torque_clip_longest_s=longest_episode(sat, dp),
                                       torque_clip_loss_peak_Nm=np.max(np.abs(f["preclip"][select, j]-f["postclip"][select, j])),
                                       torque_rms_Nm=np.sqrt(np.mean(tau**2)), torque_peak_Nm=np.abs(tau).max(),
                                       torque_rate_peak_Nm_s=np.max(np.abs(np.diff(tau)), initial=0)/dp,
                                       residual_rms_Nm=np.sqrt(np.mean(f["residual_filtered"][select, j]**2))))
    contact_rows = []
    for label, start in (("all", 0), ("final10", p["t_post"][-1]-10)):
        chosen = f["t_post"] > start + 1e-8
        for j, side in enumerate(("left", "right")):
            c = f["contacts"][chosen, j]
            valid = c[:, 0] > 0
            wheel = WHEELS[j]
            contact_rows.append(dict(run=run.name, side=side, window=label, contact_frac=valid.mean(),
                                     mean_load_N=float(c[:, 1].mean()), mean_mu=c[valid, 4].mean() if valid.any() else None,
                                     slip_long_rms_m_s=np.sqrt(np.mean(c[valid, 2]**2)) if valid.any() else None,
                                     slip_lat_rms_m_s=np.sqrt(np.mean(c[valid, 3]**2)) if valid.any() else None,
                                     wheel_desired_rad_s=np.mean(f["target"][chosen, wheel]),
                                     wheel_gt_rad_s=np.mean(qvel[chosen, wheel])))
        valid = chosen & both
        if valid.any():
            contact_rows.append(dict(run=run.name, side="both", window=label, contact_frac=both[chosen].mean(),
                                     simple_vx_rmse=np.sqrt(np.mean((y_kinematic[valid, 1]-y_kinematic[valid, 0])**2)),
                                     contact_vx_rmse=np.sqrt(np.mean((y_kinematic[valid, 2]-y_kinematic[valid, 0])**2))))
    return rows, joint_rows, phase_rows, contact_rows, frf_rows


def infer_initial(q, v):
    from math import atan2, asin

    w, x, y, z = q[3:7]
    roll = atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = asin(np.clip(2*(w*y-z*x), -1, 1))
    bx = [1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)]
    by = [2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)]
    return np.r_[np.dot(bx, v[:3]), np.dot(by, v[:3]), v[5], roll, pitch, q[2]]


def plot_summary(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.right": False, "axes.spines.top": False})
    groups = (
        ("stop_hold", ["stop_long", "stop--stop_latch_v=0.08", "stop--stop_latch_v=0.12"], (0, 9)),
        ("steps_and_ramps", ["forward_step", "forward_ramp", "left_step", "left_ramp"], (0, 5)),
        ("solver_turn", ["turn_left", "turn_left_impratio1", "turn_left_impratio10"], (0, 5, 7, 9)),
        ("multi_objective", ["curve", "pose_joint", "combined"], (0, 5, 6, 7, 9)),
        ("transitions", ["sequence", "sequence_mirror"], (0, 5, 9)),
        ("teacher_residual", ["residual_stop_short_off", "residual_stop_short_on",
                              "residual_roll_off", "residual_roll_on"], (0, 5, 6, 9)),
    )
    index_to_axis = dict(zip(GT, zip(CHANNELS, UNITS)))
    for label, cases, dim in groups:
        fig, axes = plt.subplots(len(dim), figsize=(9, 1.9*len(dim)), sharex=True)
        for name in cases:
            path = root / name / "policy.npz"
            if not path.exists():
                continue
            with np.load(path) as signal:
                for ax, j in zip(axes, dim):
                    axis, unit = index_to_axis[j]
                    line, = ax.plot(signal["t_post"], signal["body_gt"][:, j], lw=1, label=name)
                    ax.plot(signal["t_post"], signal["command_applied"][:, GT.index(j)],
                            ls="--", lw=.7, color=line.get_color(), alpha=.45)
                    ax.set_ylabel(f"{axis} [{unit}]")
                    ax.grid(alpha=.2)
        axes[0].legend(ncol=2, fontsize=8)
        axes[-1].set_xlabel("Simulation time [s]")
        fig.tight_layout()
        fig.savefig(root / f"{label}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for name in ("turn_left", "turn_left_impratio1", "turn_left_impratio10"):
        if not (root/name/"manifest.json").exists():
            continue
        _, p, f = load_traces(root/name)
        axes[0].plot(p["t_post"], p["body_gt"][:, 0], label=name)
        axes[1].plot(f["t_pre"], f["target"][:, WHEELS].mean(axis=1))
        axes[2].plot(f["t_pre"], f["contacts"][:, 1, 1]-f["contacts"][:, 0, 1], alpha=.7)
        axes[3].plot(f["t_pre"], f["contacts"][:, :, 2].mean(axis=1), alpha=.7)
    for ax, label in zip(axes, ("Body vx [m/s]", "Mean wheel target [rad/s]",
                                "Right minus left load [N]", "Mean longitudinal slip [m/s]")):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set_xlabel("Simulation time [s]")
    fig.tight_layout()
    fig.savefig(root / "turn_causal_chain.png", dpi=150)
    plt.close(fig)



def paired_tables(root, metrics, phases):
    by_index = {(r["run"], r["phase"], r["window"], r["channel"]): r for r in metrics}
    rolls = [("turn_left", "turn_left_impratio1"), ("turn_left", "turn_left_impratio10")]
    rolls.extend((f"turn_left--baseline--seed{s}", f"turn_left--impratio=1--seed{s}") for s in (42, 43, 44))
    rolls.extend((f"residual_{case}_off", f"residual_{case}_on")
                 for case in ("stop_short", "roll", "height", "stop_drive_stop"))
    rolls.extend((case, f"{case}--{change}") for case, change in (
        ("stop", "push=10"), ("stop", "sustain=10"), ("forward_step", "push=10"),
        ("forward_step", "sustain=10")))
    phase_index = {(r["run"], r["phase"]): r for r in phases}
    rows = []
    for base, altered in rolls:
        if not all((root / name / "manifest.json").exists() for name in (base, altered)):
            continue
        with np.load(root/base/"initial.npz") as a, np.load(root/altered/"initial.npz") as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        for (run, phase, window, channel), reference in by_index.items():
            if run != base or window != "final10":
                continue
            alternative = by_index.get((altered, phase, window, channel))
            if alternative is not None:
                pa, pb = phase_index[base,phase], phase_index[altered,phase]
                rows.append(dict(baseline=base, intervention=altered, phase=phase,
                                 signal=channel, unit=reference["unit"], baseline_mae=reference["mae"],
                                 intervention_mae=alternative["mae"], delta_mae=alternative["mae"]-reference["mae"],
                                 baseline_mean=reference["mean"], intervention_mean=alternative["mean"],
                                 baseline_displacement_m=pa["displacement_m"],
                                 intervention_displacement_m=pb["displacement_m"]))
    return rows


def causal_onset(root):
    _, policy, phy = load_traces(root/"turn_left")
    out = []
    for name in ("turn_left_impratio1", "turn_left_impratio10"):
        _, alt_p, alt_f = load_traces(root/name)
        for label, reference, intervention, t, threshold in (
            ("actor_gain_output", policy["action"], alt_p["action"], policy["t_infer_sim"], 1e-5),
            ("joint_target", phy["target"], alt_f["target"], phy["t_pre"], 1e-5),
            ("wheel_normal_load", phy["contacts"][:,:,1], alt_f["contacts"][:,:,1], phy["t_pre"], 1e-3),
            ("joint_torque", phy["qfrc_actuator"], alt_f["qfrc_actuator"], phy["t_pre"], 1e-4),
            ("post_step_base_vx", policy["body_gt"][:,0:1], alt_p["body_gt"][:,0:1], policy["t_post"], 1e-5),
        ):
            delta = np.max(np.abs(reference-intervention), axis=1)
            first = np.flatnonzero(delta > threshold)
            out.append(dict(case=name, signal=label, threshold=threshold,
                            first_divergence_s=t[first[0]] if first.size else None,
                            first_tick_delta=delta[0], peak_delta=delta.max()))
    return out


def no_slip_fk(root):
    """Rebuild planar sphere odometry from pose and encoder rates, without translational qvel."""
    import mujoco as mj

    model = mj.MjModel.from_xml_path(str(REPO/"sim2sim/robot/tita/mujoco_corrected/scene.xml"))
    data = mj.MjData(model)
    jacobian = np.empty((3, model.nv))
    wheel_bodies = [model.body(side + "_leg_4").id for side in ("left", "right")]
    geom = [next(i for i in range(model.ngeom) if model.geom_bodyid[i] == b
                 and model.geom_contype[i]) for b in wheel_bodies]
    rows = []
    for condition in ("forward_12", "turn_left", "turn_left_impratio1", "sequence", "combined"):
        meta, p, f = load_traces(root / condition)
        samples = []
        dt = meta["dt_policy"]
        stride = round(dt/meta["dt_physics"])
        for tick in range(len(p["tick"])):
            i = tick*stride
            if not (f["contacts"][i, :, 0] > 0).all():
                continue
            data.qpos[:] = f["qpos_pre"][i]
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            q = data.qpos[3:7]
            rotation = np.empty(9)
            mj.mju_quat2Mat(rotation, q)
            body_x = rotation.reshape(3,3)[:,0]
            rate = np.r_[p["obs"][tick,:3] / .25, f["sensor_dq"][i]]
            predicted = []
            for body, g in zip(wheel_bodies, geom):
                point = data.geom_xpos[g] - [0., 0., .0925]
                mj.mj_jac(model, data, jacobian, None, point, body)
                # Translational columns are deliberately excluded from the estimate.
                predicted.append(-body_x @ (jacobian[:,3:] @ rate))
            gt = body_x @ f["qvel_pre"][i,:3]
            simple = .0925*f["sensor_dq"][i, list(WHEELS)].mean()
            samples.append([p["t_post"][tick]-dt, gt, simple, *predicted])
        merged = np.asarray(samples)
        np.savez_compressed(root/condition/"odometry.npz", values=merged,
                            columns=["time_s","base_vx_gt","wheel_only_vx","left_fk_vx","right_fk_vx"])
        for label, start in (("full", 0), ("final10", p["t_post"][-1]-10)):
            selected = merged[merged[:,0] >= start - 1e-8]
            for estimator, values in (("wheel_only", selected[:,2]),
                                      ("articulated_fk", selected[:,3:].mean(axis=1))):
                rows.append(dict(case=condition, window=label, estimator=estimator, n=len(selected),
                                 **tracking(values-selected[:,1])))
    return rows


def recovery(root):
    rows = []
    for name in ("stop", "forward_step"):
        _, base, _ = load_traces(root/name)
        for variant, release in (("push", 5.10), ("sustain", 7.)):
            run = root/f"{name}--{variant}=10"
            if not (run/"manifest.json").exists():
                continue
            manifest, p, f = load_traces(run)
            t = p["t_post"]
            remaining = t >= release
            idx = np.searchsorted(base["t_post"], t)
            idx = np.minimum(idx, len(base["t_post"])-1)
            delta = p["body_gt"] - base["body_gt"][idx]
            dims = [0, 1, 5, 6, 7, 9]
            steady = np.abs(delta[:,dims]) <= np.array([.02,.02,.03,np.deg2rad(1),np.deg2rad(1),.005])
            stable = remaining & steady.all(axis=1)
            dt = manifest["dt_policy"]
            event = np.flatnonzero(np.convolve(stable.astype(int), np.ones(round(1/dt), dtype=int), "valid")
                                   == round(1/dt))
            mask = t > 5
            rows.append(dict(case=name, intervention=variant, impulse_N_s=np.sum(f["external_world_y_N"])*manifest["dt_physics"],
                             recovery_s=max(0.,t[event[0]]-release) if event.size else None,
                             observation_after_release_s=t[-1]-release,
                             roll_peak_deg=np.rad2deg(np.abs(delta[mask,6]).max()),
                             pitch_peak_deg=np.rad2deg(np.abs(delta[mask,7]).max()),
                             height_peak_m=np.abs(delta[mask,9]).max()))
    return rows


def residual_gate_summary(root):
    rows = []
    for run in sorted(root.glob("residual_*/manifest.json")):
        meta, p, f = load_traces(run.parent)
        nonzero = np.any(p["command_applied"][:, :3] != 0, axis=1)[f["policy_tick"]]
        active = f["residual_active"]
        requested, filtered = f["residual_requested"], f["residual_filtered"]
        assert not active[nonzero].any()
        assert not requested[~active].any() and not filtered[~active].any()
        if meta["residual"] == "off":
            assert not active.any()
        if meta["residual"] == "on":
            assert active.any()
        assert meta["residual_filter_hz"] == 60 and meta["dt_physics"] == .002
        assert np.max(np.abs(requested)) <= 10
        rows.append(dict(run=run.parent.name, active_fraction=active.mean(), moving_ticks=int(nonzero.sum()),
                         request_peak_nm=np.max(np.abs(requested)), filter_peak_nm=np.max(np.abs(filtered)),
                         all_joint_torque_rms_nm=np.sqrt(np.mean(f["qfrc_actuator"]**2)),
                         saturation_fraction=np.mean(np.abs(f["preclip"]-f["postclip"]) > 1e-5)))
    return rows


def tracking_reward_terms(root):
    rows = []
    for name in ("stop", "forward_12", "turn_left", "curve", "pose_joint", "combined"):
        _, p, _ = load_traces(root/name)
        sl = p["t_post"] > p["t_post"][-1]-10
        y = p["body_gt"][sl][:,GT]
        command = p["command_applied"][sl]
        err = y-command
        roll, pitch = y[:,3], y[:,4]
        roll_cmd, pitch_cmd = command[:,3], command[:,4]
        upright = np.clip(np.cos(roll)*np.cos(pitch), 0., .7)/.7
        xy_sq = np.square(err[:,:2]).sum(1)
        lin = sum(weight*np.exp(-xy_sq/variance) for weight, variance in ((2.,.25),(.5,.0625),(.5,.03125)))
        yaw = np.exp(-err[:,2]**2/.25)
        gravity_sq = (np.sin(pitch)-np.sin(pitch_cmd))**2
        gravity_sq += (np.sin(roll)*np.cos(pitch)-np.sin(roll_cmd)*np.cos(pitch_cmd))**2
        rows.append(dict(case=name, linear=np.mean(lin*upright), yaw=np.mean(yaw*upright),
                         orientation=np.mean(-5*gravity_sq*upright), height=np.mean(-10*err[:,5]**2*upright)))
    return rows


def plot_frf(root, rows):
    import matplotlib.pyplot as plt

    fig, (magnitude, phase) = plt.subplots(2, 1, figsize=(7.5, 5.3), sharex=True)
    for signal in ("vx", "wz"):
        values = sorted((row for row in rows if row["signal"] == signal), key=lambda row: row["hz"])
        freq = [row["hz"] for row in values]
        gain = np.array([row["gain"] for row in values])
        angle = [row["phase_deg"] for row in values]
        magnitude.semilogx(freq, 20*np.log10(gain), marker="o", label=signal)
        phase.semilogx(freq, angle, marker="o", label=signal)
    magnitude.set_ylabel("Fundamental gain [dB]")
    phase.set_ylabel("Wrapped phase [deg]")
    phase.set_xlabel("Frequency [Hz]")
    magnitude.legend()
    for ax in (magnitude, phase):
        ax.grid(which="both", alpha=.25)
    fig.tight_layout()
    fig.savefig(root/"frequency_response.png", dpi=150)
    plt.close(fig)


def verify_trace_contract(root):
    _, base_p, base_f = load_traces(root/"left_step")
    _, clock_p, clock_f = load_traces(root/"clock_50-200-200")
    for key in ("obs", "action", "qpos", "qvel"):
        np.testing.assert_array_equal(base_p[key], clock_p[key])
    for key in ("target", "ctrl", "qfrc_actuator"):
        np.testing.assert_array_equal(base_f[key], clock_f[key])
    for name in ("forward_12", "turn_left", "sequence", "combined"):
        meta, p, f = load_traces(root/name)
        np.testing.assert_allclose(f["t_post"]-f["t_pre"], meta["dt_physics"], atol=1e-10)
        np.testing.assert_allclose(f["postclip"], np.clip(f["preclip"], -np.array(meta["torque_limit"]),
                                                          meta["torque_limit"]), atol=1e-5)
        np.testing.assert_array_equal(f["ctrl"], f["qfrc_actuator"])
        np.testing.assert_allclose(p["obs"][:,6:9], p["command_applied"][:,:3]*[2,2,.25], atol=1e-6)
    # Synthetic injection checks for the derived metrics.
    t = np.arange(0, 4, .01)
    a, phase, residual = fits(t, .4*np.sin(2*np.pi*2*t-.6), 2)
    np.testing.assert_allclose([a, phase, residual], [.4, -.6, 0.], atol=1e-12)
    assert longest_episode([True, True, False, True], .005) == .01
    print("[DESKTOP_DRY_RUN] Clock baseline, tick alignment, observation scale, joint torque, residual gate and metric checks passed")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--archive-only", action="store_true")
    args = parser.parse_args()
    write_csv(args.root / "heading_archive.csv", archived_trajectory())
    if args.archive_only:
        return
    outputs = ([], [], [], [], [])
    for manifest in sorted(args.root.glob("*/manifest.json")):
        if json.loads(manifest.read_text())["execution_domain"] != "VALIDATION_SIM":
            continue
        for rows, result in zip(outputs, characterize(manifest.parent)):
            rows.extend(result)
    for name, rows in zip(("metrics", "joint_metrics", "transitions", "contact_metrics", "frequency_response"), outputs):
        if rows:
            write_csv(args.root / f"{name}.csv", rows)
    pairs = paired_tables(args.root, outputs[0], outputs[2])
    write_csv(args.root / "paired_effects.csv", pairs)
    seed_rows = []
    for channel, unit in zip(CHANNELS, UNITS):
        trials = [r for r in pairs if "--seed" in r["baseline"] and r["signal"] == channel]
        if trials:
            delta = np.array([r["delta_mae"] for r in trials])
            worst = max(trials, key=lambda r: r["intervention_mae"])
            seed_rows.append(dict(signal=channel, unit=unit, n=len(trials),
                                  paired_delta_mean=delta.mean(), paired_delta_std=delta.std(ddof=1),
                                  worst_intervention_mae=worst["intervention_mae"], worst_run=worst["intervention"]))
    if seed_rows:
        write_csv(args.root / "seed_effects.csv", seed_rows)
    manifests = [json.loads(path.read_text()) | {"run":path.parent.name}
                 for path in sorted(args.root.glob("*/manifest.json"))]
    fields = ("run", "execution_domain", "evidence_status", "case", "variant", "seed", "residual", "ticks",
              "dt_policy", "dt_pd", "dt_physics", "wall_seconds", "checkpoint_sha256")
    write_csv(args.root / "run_index.csv", [{key: item.get(key) for key in fields} for item in manifests])
    write_csv(args.root / "causal_onset.csv", causal_onset(args.root))
    write_csv(args.root / "odometry.csv", no_slip_fk(args.root))
    write_csv(args.root / "recovery.csv", recovery(args.root))
    write_csv(args.root / "residual_gate.csv", residual_gate_summary(args.root))
    write_csv(args.root / "tracking_reward_terms.csv", tracking_reward_terms(args.root))
    verify_trace_contract(args.root)
    plot_summary(args.root)
    if outputs[4]:
        plot_frf(args.root, outputs[4])


if __name__ == "__main__":
    main()
