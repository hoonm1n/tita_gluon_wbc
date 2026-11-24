"""VALIDATION_SIM and TRAINING_SIM evaluation of the frozen TITA controllers."""

import argparse
from collections import defaultdict, deque
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from evaluate_sim2sim_heading import POSE, REPO, collect_sample, run_scenario

CKPT = REPO / "mj_lab/logs/flash_sac/heading_baseline/model_85000"
OLD = REPO / "docs/exp/2026-11-20"
ORIGINAL = REPO.parent / "tita_gluon_wbc"
STORE = REPO / "sim2sim/checkpoints/mjlab/task/base_velocity/height_roll_pitch"
TEACHER = STORE / "20261108/80000"
RESIDUAL = STORE / "20261109/10000_d6672529ee24/residual.policy.pt"
PACKAGE = REPO / "sim2sim/sim2sim_ros2_jazzy"
STOP = (0., 0.)

# Segment format: duration (s), target (vx, wz), roll/pitch/height, speed ramp (s).
def segment(duration, speed=STOP, pose=POSE, ramp=0.):
    return duration, speed, pose, ramp


def scenarios():
    cases = {}
    for name, speed in dict(forward=(.8, 0), reverse=(-.8, 0),
                            left=(0, 1), right=(0, -1)).items():
        cases[name + "_step"] = [segment(20, speed)]
        cases[name + "_ramp"] = [segment(20, speed, ramp=1)]
    cases.update(
        forward_12=[segment(60, (1.2, 0))],
        turn_left=[segment(60, (0, 1))],
        stop=[segment(60)],
        stop_long=[segment(120)],
        roll=[segment(30, pose=(np.deg2rad(10), 0, .335))],
        pitch=[segment(60, pose=(0, np.deg2rad(5), .335))],
        height=[segment(30, pose=(0, 0, .370))],
        height_up=[segment(60, pose=(0, 0, .370))],
        stop_short=[segment(30)],
        stop_drive_stop=[segment(8), segment(8, (.4, 0)), segment(20)],
        curve=[segment(60, (.8, .8))],
        curve_height=[segment(60, (.8, .8), (0, 0, .370))],
        pose_joint=[segment(60, pose=(np.deg2rad(10), np.deg2rad(5), .370))],
        combined=[segment(60, (.8, .8), (np.deg2rad(10), np.deg2rad(5), .370))],
        sequence=[segment(8, cmd) for cmd in (STOP, (.8, 0), (-.8, 0), (.8, .8), (0, 1))]
        + [segment(20)],
        sequence_mirror=[segment(8, cmd) for cmd in (STOP, (-.8, 0), (.8, 0), (.8, -.8), (0, -1))]
        + [segment(20)],
        forward_limit=[segment(20, (1.5, 0))],
        reverse_limit=[segment(20, (-1.5, 0))],
        yaw_limit=[segment(20, (0, 3))],
        yaw_limit_mirror=[segment(20, (0, -3))],
    )
    return cases


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git_provenance(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    return dict(commit=git("rev-parse", "HEAD").decode().strip(),
                diff_sha256=hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest())


def verify_bundle(root, teacher=False):
    config_dir = root if teacher else root / "sim2sim/actor"
    manifest = json.loads((config_dir / "manifest.json").read_text())
    actor = root / ("policy.pt" if teacher else "actor.pt")
    for key, path in (("checkpoint_sha256", actor),
                      ("onnx_sha256", config_dir / "policy.onnx"),
                      ("config_sha256", config_dir / "config.json")):
        if sha256(path) != manifest[key]:
            raise ValueError(f"Invalid {key} for {path}")
    return config_dir / "config.json", manifest


class Trace:
    def __init__(self):
        self.columns = defaultdict(list)

    def append(self, **row):
        for key, value in row.items():
            self.columns[key].append(np.asarray(value).copy())

    def save(self, path):
        np.savez_compressed(path, **{key: np.array(value) for key, value in self.columns.items()})


def contact_sample(mj, model, data, floor, wheels, pre_vel, x_axis):
    """Sample contact forces and material point velocities from one solver state."""
    jac = np.empty((3, model.nv))
    force = np.empty(6)
    wheel = np.zeros((2, 11))
    body_contact = np.zeros(2)
    body_names = ("base_link", "left_leg_3", "right_leg_3")
    for contact_id, contact in enumerate(data.contact):
        if floor not in contact.geom or contact.efc_address < 0:
            continue
        other = contact.geom[1] if contact.geom[0] == floor else contact.geom[0]
        body = model.geom_bodyid[other]
        mj.mj_contactForce(model, data, contact_id, force)
        if model.body(body).name in body_names:
            slot = int(model.body(body).name != "base_link")
            body_contact[slot] += np.linalg.norm(force[:3])
        if body not in wheels:
            continue
        i = wheels.index(body)
        normal = contact.frame.reshape(3, 3)[0]
        longitudinal = x_axis - normal * (normal @ x_axis)
        longitudinal /= np.linalg.norm(longitudinal)
        lateral = np.cross(normal if normal[2] > 0 else -normal, longitudinal)
        mj.mj_jac(model, data, jac, None, contact.pos, body)
        velocity = jac @ pre_vel
        wheel[i] += (1, force[0], longitudinal @ velocity, lateral @ velocity,
                     contact.friction[0], *contact.pos, *normal)
    active = wheel[:, 0] > 0
    wheel[active, 2:] /= wheel[active, :1]
    return wheel, body_contact


def set_condition(node, variant):
    env, ad = node.env, node.action_adapter
    model, data, mj = env.model, env.data, env.mujoco
    key, _, value = variant.partition("=")
    if key == "baseline":
        return
    numeric = float(value)
    if key == "cylinder":
        for side in ("left", "right"):
            body = model.body(side + "_leg_4").id
            ids = np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0))
            if len(ids) != 1:
                raise ValueError("Expected exactly one wheel geom")
            geom = ids[0]
            model.geom_type[geom] = mj.mjtGeom.mjGEOM_CYLINDER
            model.geom_size[geom] = [.0925, .017, 0.]
            model.geom_rbound[geom] = np.hypot(.0925, .017)
            model.geom_aabb[geom] = [0., 0., 0., .0925, .0925, .017]
        mj.mj_forward(model, data)
    elif key == "impratio":
        model.opt.impratio = numeric
    elif key == "iterations":
        model.opt.iterations = int(numeric)
    elif key == "line_search":
        model.opt.ls_iterations = int(numeric)
    elif key == "mu":
        env.set_ground_friction(numeric)
        model.geom_friction[:, 0] = numeric
    elif key == "stop_latch_v":
        node.obs_builder.position_term["params"]["velocity_thresholds"][0] = numeric
    elif key == "leg_kp":
        ad.kp[~ad.velocity] *= numeric
    elif key == "leg_kd":
        ad.kd[~ad.velocity] *= numeric
    elif key == "wheel_kd":
        ad.kd[ad.velocity] *= numeric
    elif key == "torque_delay":
        env.set_torque_delay(numeric, numeric)
    elif key == "torque_jitter":
        env.set_torque_delay(numeric - 5, numeric + 5)
    elif key == "gyro_noise":
        env.set_sensor_noise(gyro=numeric)
    elif key in ("mass", "com_x", "com_y", "com_z", "inertia"):
        base = model.body("base_link").id
        if key == "mass":
            model.body_mass[base] += numeric
        elif key == "inertia":
            model.body_inertia[base] *= numeric
        else:
            model.body_ipos[base, "xyz".index(key[-1])] += numeric
        state = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), float(data.time)
        mj.mj_setConst(model, data)
        data.qpos[:], data.qvel[:], data.ctrl[:], data.time = state
        data.qacc_warmstart[:] = 0
        mj.mj_forward(model, data)
    elif key not in ("push", "sustain", "gyro_bias", "gyro_dropout", "sine_vx", "sine_wz",
                    "action_delay", "actor_sensor_delay", "pd_sensor_delay"):
        raise ValueError(f"Unknown experimental condition {variant}")


def prepare_node(node):
    dt = node.env.control_dt
    for _ in range(round(5 / dt)):
        node._step()
        if node.sm.state == "rl":
            break
    else:
        raise RuntimeError("Sim2Sim did not enter RL")
    for _ in range(round(1 / dt)):
        node._step()


def drive(node, t, segments, variant, previous, phase):
    from geometry_msgs.msg import Twist, Vector3

    duration, speed, pose, ramp = segments[phase]
    blend = min(t / ramp, 1) if ramp else 1
    vx, wz = np.asarray(previous) + blend * (np.asarray(speed) - previous)
    key, _, value = variant.partition("=")
    if key.startswith("sine_"):
        oscillation = np.sin(2 * np.pi * float(value) * t)
        vx += .1 * oscillation if key == "sine_vx" else 0
        wz += .15 * oscillation if key == "sine_wz" else 0
    command = Twist()
    command.linear.x, command.angular.z = float(vx), float(wz)
    node._cmd_twist_cb(command)
    if t == 0:
        node._cmd_pose_cb(Vector3(x=float(pose[0]), y=float(pose[1]), z=float(pose[2])))


def sim_run(out, name, variant, residual, seed, parity, initial_roll=0., clocks=None):
    sys.path.insert(0, str(PACKAGE))
    import rclpy
    from sim2sim_ros2_jazzy.adapter import NATIVE_JOINTS, quat_wxyz_to_rotmat, quat_wxyz_to_rpy
    from sim2sim_ros2_jazzy.sim2sim_node import Sim2SimNode

    root = CKPT if residual is None else TEACHER
    saved_config, hashes = verify_bundle(root, teacher=residual is not None)
    config = json.loads(saved_config.read_text())
    # Published Teacher paths are from the export machine. Keep the verified bytes.
    if residual is not None:
        config["policy"]["path"] = str(TEACHER / "policy.onnx")
        config["policy_interface"]["checkpoint"] = str(TEACHER / "policy.pt")
    out.mkdir(parents=True, exist_ok=False)
    effective_config = out / "config.json"
    output_json(effective_config, config)
    args = ["--ros-args", "-p", f"config:={effective_config}", "-p", "viewer:=false"]
    if residual is not None:
        expected = json.loads((RESIDUAL.parent / "residual/manifest.json").read_text())
        if sha256(RESIDUAL) != expected["checkpoint_sha256"] or (
                hashes["checkpoint_sha256"] != expected["teacher_checkpoint_sha256"]):
            raise ValueError("Teacher/residual bundle mismatch")
        args += ["-p", f"learned_residual:={RESIDUAL}"]
    reference = []
    if parity:
        duration, (vx, wz), _, _ = scenarios()[name][0]
        run_scenario(effective_config, (vx, 0., wz), duration, reference)
    rclpy.init(args=args)
    node = None
    started = time.monotonic()
    try:
        node = Sim2SimNode()
        env, adapter = node.env, node.action_adapter
        mj, m, d = env.mujoco, env.model, env.data
        actuator_ids = np.asarray(env.actuator_ids)
        wheel_ids = [m.body(side + "_leg_4").id for side in ("left", "right")]
        wheel_geoms = [next(i for i in range(m.ngeom) if m.geom_bodyid[i] == body
                            and (m.geom_contype[i] or m.geom_conaffinity[i])) for body in wheel_ids]
        floor = m.geom("floor").id
        rotation = quat_wxyz_to_rotmat
        rpy = quat_wxyz_to_rpy
        if residual is not None:
            node.disturbance.residual_enabled = False
        prepare_node(node)
        if initial_roll:
            original_quaternion = d.qpos[3:7].copy()
            half_roll = np.deg2rad(initial_roll) / 2
            mj.mju_mulQuat(d.qpos[3:7], original_quaternion,
                           np.array([np.cos(half_roll), np.sin(half_roll), 0., 0.]))
            mj.mj_forward(m, d)
        if residual is not None:
            node.disturbance.residual_enabled = residual == "on"
        env.sensor_rng = np.random.default_rng(seed)
        env.delay_rng = np.random.default_rng(seed + 1)
        for change in variant.split(","):
            set_condition(node, change)
        if clocks:
            policy_hz, pd_hz, physics_hz = clocks
            if physics_hz % policy_hz or physics_hz % pd_hz or pd_hz % policy_hz or residual is not None:
                raise ValueError("Require nested frequencies and the base policy for clock experiments")
            m.opt.timestep = 1 / physics_hz
            env.control_dt = 1 / policy_hz
            env.decimation = physics_hz // policy_hz
            node.obs_builder.dt = env.control_dt
        if variant.startswith("sine_"):
            hold = scenarios()[name][0]
            drive(node, 0., [hold], "baseline", STOP, 0)
            for _ in range(round(20/env.control_dt)):
                node._step()
        start = float(d.time)
        for body in wheel_ids:
            assert np.dot(np.cross(d.xaxis[m.body_jntadr[body]], [0, 0, 1]),
                          rotation(d.qpos[3:7])[:, 0]) > .8
        np.savez_compressed(out / "initial.npz", qpos=d.qpos.copy(), qvel=d.qvel.copy(),
                            last_action=node.last_action.copy(), ctrl=d.ctrl.copy())
        meta = json.loads((root / "ddt.json").read_text())
        pos_gain = np.asarray(meta["action_gain_positive"])
        neg_gain = np.asarray(meta["action_gain_negative"])
        policy, physics = Trace(), Trace()
        policy_tick, phase, segment_time = 0, 0, 0.
        infer, step, compute = node.policy.run, mj.mj_step, adapter.compute
        gyro = env.base_ang_vel
        gyro_history = deque(maxlen=1)
        key, _, amount = variant.partition("=") if "," not in variant else ("combined", "", "")
        amount = float(amount) if amount else 0.
        lag = round(amount / (1000 * m.opt.timestep)) if key.endswith("_delay") else 0
        action_history = deque([node.last_action.copy()] * lag, maxlen=lag+1)
        sensor_history = deque(maxlen=lag+1)
        build = node.obs_builder.build
        events = []
        base_contact_history = deque(maxlen=3)
        stopped = False

        def record_policy(outputs, inputs):
            nonlocal infer_row
            before = time.monotonic_ns()
            value = infer(outputs, inputs)
            after = time.monotonic_ns()
            action = value[0].ravel()
            observation = inputs["nn_input0"].ravel()
            infer_row = dict(obs=observation.copy(), action=action.copy(),
                             normalized_action=action / np.where(action >= 0, pos_gain, neg_gain),
                             infer_begin_ns=before, infer_end_ns=after, t_infer_sim=d.time - start)
            return value

        def record_pd(action, positions, velocities, gait_enabled=None, residual=None):
            nonlocal pd_row
            raw_action = action.copy()
            if key == "action_delay" and lag:
                action_history.append(action.copy())
                action = action_history[0]
            if key == "pd_sensor_delay" and lag and sensor_history:
                positions, velocities = sensor_history[max(0, len(sensor_history)-lag)][:2]
            result = compute(action, positions, velocities, gait_enabled=gait_enabled, residual=residual)
            target = adapter.targets(action, gait_enabled)
            before_clip = np.empty(8)
            before_clip[adapter.order] = action * adapter.scale + adapter.offset
            p = adapter.kp * np.where(adapter.velocity, 0., target - positions)
            derivative = adapter.kd * (np.where(adapter.velocity, target, 0.) - velocities)
            filtered = np.zeros(8) if residual is None else residual.copy()
            requested = np.zeros(8) if node.learned is None else node.learned.residual.copy()
            pd_row = dict(action_pd=action.copy(), action_before_delay=raw_action, sensor_q=positions.copy(), sensor_dq=velocities.copy(),
                          target_raw=before_clip, target=target, p=p, d=derivative, pd=p + derivative,
                          preclip=p + derivative + filtered, postclip=result.copy(),
                          residual_requested=requested, residual_filtered=filtered,
                          residual_active=bool(node.learned is not None and node.learned.enabled),
                          t_pd_sim=d.time - start, pd_ns=time.monotonic_ns())
            return result

        def record_step(model, data):
            nonlocal stopped
            if key in ("actor_sensor_delay", "pd_sensor_delay"):
                sensor_history.append((env.joint_pos(), env.joint_vel(),
                                       env.base_quat_sensor(), env.base_ang_vel(), env.base_lin_vel()))
            pre_pos, pre_vel = d.qpos.copy(), d.qvel.copy()
            true_r = rotation(pre_pos[3:7])
            pre_t = float(d.time) - start
            duration = .10 if key == "push" else 2.
            force = amount if key in ("push", "sustain") and 5 <= pre_t + 1e-8 < 5 + duration else 0.
            d.xfrc_applied[m.body("base_link").id, 1] = force
            apply_ns = time.monotonic_ns()
            step(model, data)
            contacts, illegal = contact_sample(mj, m, d, floor, wheel_ids, pre_vel, true_r[:, 0])
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                raise FloatingPointError("Non-finite MuJoCo state")
            base_contact_history.append(float(illegal[0]))
            if max(base_contact_history) > 1 and not stopped:
                events.append((pre_t, "base_contact_gt_1N", float(illegal[0])))
                stopped = True
            physics.append(
                policy_tick=policy_tick, physics_tick=len(physics.columns["t_pre"]),
                t_pre=pre_t, t_post=d.time - start, qpos_pre=pre_pos, qvel_pre=pre_vel,
                qpos_post=d.qpos, qvel_post=d.qvel, ctrl=d.ctrl[actuator_ids],
                actuator_force=d.actuator_force[actuator_ids],
                qfrc_actuator=d.qfrc_actuator[env.qvel_addr], delay_ms=env.last_delay_ms,
                contacts=contacts, nonwheel_load=illegal, external_world_y_N=force,
                wheel_centers_world=d.geom_xpos[wheel_geoms], apply_ns=apply_ns, **pd_row,
            )
            for body in wheel_ids:
                if not np.isfinite(d.xpos[body]).all():
                    raise FloatingPointError("Non-finite contact kinematics")

        def gyro_with_impairment():
            sample = gyro()
            if key == "gyro_bias":
                sample[2] += amount
            if key == "gyro_dropout" and 50 <= policy_tick % 250 < 50 + int(amount) and gyro_history:
                return gyro_history[-1].copy()
            gyro_history.append(sample.copy())
            return sample

        def build_with_delay(state):
            if sensor_history:
                q, dq, quat, w, v_world = sensor_history[max(0, len(sensor_history)-lag)]
                state = replace(state, joint_pos=q, joint_vel=dq, base_quat_wxyz=quat,
                                base_ang_vel=w, base_lin_vel_b=rotation(quat).T @ v_world)
            return build(state)

        def step_clocks(callback):
            ticks_per_pd = clocks[2] // clocks[1]
            for tick in range(env.decimation):
                if tick % ticks_per_pd == 0:
                    applied_torque = callback(env.joint_pos(), env.joint_vel())
                d.ctrl[actuator_ids] = applied_torque
                mj.mj_step(m, d)

        segments = scenarios()[name]
        if key.startswith("sine_"):
            duration, speed, pose, ramp = segments[0]
            segments = [segment(8 / amount, speed, pose, ramp)]
        baseline_samples = []
        previous = STOP
        request_start = time.monotonic_ns()
        with ExitStack() as patches:
            patches.enter_context(patch.object(node.policy, "run", record_policy))
            patches.enter_context(patch.object(mj, "mj_step", record_step))
            patches.enter_context(patch.object(adapter, "compute", record_pd))
            if key in ("gyro_bias", "gyro_dropout"):
                patches.enter_context(patch.object(env, "base_ang_vel", gyro_with_impairment))
            if key == "actor_sensor_delay" and lag:
                patches.enter_context(patch.object(node.obs_builder, "build", build_with_delay))
            if clocks:
                patches.enter_context(patch.object(env, "step_with_torque_callback", step_clocks))
            for phase, (duration, speed, pose, ramp) in enumerate(segments):
                for local_tick in range(round(duration / env.control_dt)):
                    segment_time = local_tick * env.control_dt
                    request_ns = time.monotonic_ns()
                    drive(node, segment_time, segments, variant, previous, phase)
                    received_ns = time.monotonic_ns()
                    applied = node.command.copy()
                    applied_pose = node.pose_command.copy()
                    requested = node.requested_command.copy()
                    pre_ns = time.monotonic_ns()
                    infer_row, pd_row = {}, {}
                    node._step()
                    if not infer_row or node.sm.state != "rl":
                        events.append((d.time - start, "left_rl", 0.))
                        stopped = True
                        break
                    base_rotation = rotation(d.qpos[3:7])
                    previous_post_gt = np.r_[base_rotation.T @ d.qvel[:3], d.qvel[3:6],
                                             rpy(d.qpos[3:7]), d.qpos[2]]
                    policy.append(
                        tick=policy_tick, phase=phase, phase_time=segment_time, t_post=d.time-start,
                        request_ns=request_ns, received_ns=received_ns, step_begin_ns=pre_ns,
                        step_end_ns=time.monotonic_ns(), requested=np.r_[requested, pose],
                        command_limited=np.r_[node.command_target, node.pose_target],
                        command_applied=np.r_[applied, applied_pose],
                        scheduled_speed=np.r_[speed[0], 0., speed[1]], body_gt=previous_post_gt,
                        body_gyro_sensor=env.base_ang_vel(), qpos=d.qpos, qvel=d.qvel,
                        position_latched=node.obs_builder.position_latched,
                        heading_integral=node.obs_builder.heading, **infer_row,
                    )
                    if parity:
                        row = collect_sample(node, (policy_tick + 1) * env.control_dt, applied)
                        baseline_samples.append(tuple(row.values()))
                    policy_tick += 1
                    if stopped:
                        break
                previous = speed
                if stopped:
                    break
        if parity:
            np.testing.assert_array_equal(np.asarray(baseline_samples),
                                          np.asarray([tuple(row.values()) for row in reference[1:]]))
            old_path = OLD / "sim2sim_speed_tracking_60s" / (name + ".csv")
            old = np.genfromtxt(old_path, delimiter=",", names=True)[1:1+len(baseline_samples)]
            current = np.array(baseline_samples)
            fields = list(reference[1])
            for field in ("actual_vx_m_s", "actual_vy_m_s", "actual_yaw_rate_rad_s"):
                np.testing.assert_array_equal(current[:, fields.index(field)], old[field])
        with (out / "events.csv").open("w") as stream:
            stream.write("time_s,reason,value\n")
            for event in events:
                stream.write(",".join(map(str, event)) + "\n")
        for label, trace in (("policy", policy), ("physics", physics)):
            trace.save(out / (label + ".npz"))
        output_json(out / "manifest.json", dict(
            execution_domain="VALIDATION_SIM", evidence_status="SUSPICIOUS" if stopped else "PASS",
            case=name, variant=variant, seed=seed, initial_roll_deg=initial_roll, residual=residual,
            residual_sha256=None if residual is None else sha256(RESIDUAL),
            residual_filter_hz=None if node.learned is None else node.learned.filter_hz,
            checkpoint_sha256=hashes["checkpoint_sha256"], onnx_sha256=hashes["onnx_sha256"],
            saved_config_sha256=hashes["config_sha256"], effective_config_sha256=sha256(effective_config),
            git=git_provenance(REPO), executable=sys.executable, mujoco=mj.__version__,
            xml_sha256={p.name:sha256(p) for p in (REPO / config["model"]["scene_xml"]).parent.glob("*.xml")},
            dt_physics=m.opt.timestep, dt_policy=env.control_dt, dt_pd=m.opt.timestep if clocks is None else 1/clocks[1],
            contact_columns=["count", "Fn_N", "slip_long_m_s", "slip_lat_m_s", "mu",
                             "position_x_m", "position_y_m", "position_z_m", "normal_x", "normal_y", "normal_z"],
            gt_columns=["vx_b", "vy_b", "vz_b", "wx_b", "wy_b", "wz_b", "roll", "pitch", "yaw", "height"],
            joints=list(NATIVE_JOINTS), gain_kp=adapter.kp.tolist(), gain_kd=adapter.kd.tolist(),
            torque_limit=adapter.limit.tolist(), solver=int(m.opt.solver), integrator=int(m.opt.integrator),
            impratio=float(m.opt.impratio), iterations=int(m.opt.iterations),
            ls_iterations=int(m.opt.ls_iterations), orientation="world z-up, qpos wxyz",
            mass=m.body_mass.tolist(), inertia=m.body_inertia.tolist(), com=m.body_ipos.tolist(),
            armature=m.dof_armature.tolist(), geom_friction=m.geom_friction.tolist(),
            geom_dim=m.geom_condim.tolist(), geom_priority=m.geom_priority.tolist(),
            geom_solref=m.geom_solref.tolist(), geom_solimp=m.geom_solimp.tolist(),
            segments=segments, requested_seconds=float(sum(row[0] for row in segments)),
            ticks=policy_tick, physics_ticks=len(physics.columns["t_pre"]),
            started_monotonic_ns=request_start, wall_seconds=time.monotonic()-started, parity=parity,
        ))
        print(f"[VALIDATION_SIM] {out.name} {policy_tick} ticks, events={events}", flush=True)
    finally:
        if node is not None:
            node.env.close()
            node.destroy_node()
        rclpy.try_shutdown()



def native_run(out):
    """Use the training worktree modules and the verified Sim2Sim initial states."""
    import pickle

    sys.path.insert(0, str(ORIGINAL))
    import torch
    from mj_lab.flash_sac import runtime
    from mj_lab.robots.tita.common import mdp, robot_cfg

    runtime.setup_imports(False)
    from export import load_policy
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply

    names = ("forward_12", "turn_left", "sequence")
    checkpoint, meta = runtime.read_checkpoint(CKPT, task="flat_roll_pitch_height")
    for path, expected in meta["training_source_sha256"].items():
        if sha256(ORIGINAL / path) != expected:
            raise ValueError(f"Training source mismatch for {path}")
    assert Path(mdp.__file__).is_relative_to(ORIGINAL)
    root = out.parent
    pairs = [np.load(root / name / "initial.npz") for name in names]
    references = [np.load(root / name / "policy.npz") for name in names]
    bundle = load_policy(checkpoint, meta).cuda()

    def tensor(value):
        return torch.as_tensor(np.asarray(value), device="cuda", dtype=torch.float32)

    def action(obs):
        normalized = bundle.actor.get_mean_and_std(obs, training=False)[0].tanh()
        gain = torch.where(normalized >= 0, bundle.action_gain_positive, bundle.action_gain_negative)
        return normalized, normalized * gain

    with torch.inference_mode():
        same_obs = np.concatenate([trace["obs"] for trace in references])
        same_action = np.concatenate([trace["action"] for trace in references])
        exported = action(tensor(same_obs))[1].cpu().numpy()
    np.testing.assert_allclose(exported, same_action, rtol=1e-4, atol=1e-5)
    out.mkdir(parents=True, exist_ok=False)
    with (checkpoint / "env.pkl").open("rb") as stream:
        cfg = pickle.load(stream)
    cfg.seed, cfg.scene.num_envs, cfg.episode_length_s = 42, len(names), 65.
    cfg.auto_reset = False
    friction = cfg.events["physics_material"]
    friction.params["ranges"] = (.8, .8)
    cfg.events = {"physics_material": friction}
    for motor in cfg.scene.entities["robot"].articulation.actuators:
        motor.delay_min_lag = motor.delay_max_lag = 0
    cfg.observations["policy"].enable_corruption = False
    cfg.commands["base_velocity"].sensor_bias = False
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        run_native_ticks(env, names, pairs, references, action, tensor, euler_xyz_from_quat, quat_apply,
                         robot_cfg.JOINTS, out, meta, float(np.abs(exported-same_action).max()))
    finally:
        env.close()


def run_native_ticks(env, names, pairs, references, action, tensor, euler_xyz, quat_apply,
                     joints, out, meta, onnx_delta):
    import torch

    policy, physics = Trace(), Trace()
    env.reset()
    robot = env.scene["robot"]
    data = robot.data
    vel = env.command_manager.get_term("base_velocity")
    pose = env.command_manager.get_term("base_pose")
    vel.time_left[:] = pose.time_left[:] = 1e9
    vel.gyro_bias.zero_()
    vel.radius_scale.fill_(1)
    pose.target[:] = pose.command[:] = tensor(POSE)
    qpos = tensor([z["qpos"] for z in pairs])
    qvel = tensor([z["qvel"] for z in pairs])
    root = qpos[:, :7].clone()
    root[:, :3] += env.scene.env_origins
    robot.write_root_link_pose_to_sim(root)
    robot.write_root_link_velocity_to_sim(torch.cat((qvel[:, :3], quat_apply(root[:, 3:], qvel[:, 3:6])), dim=1))
    indices = [joints.index(joint) for joint in robot.joint_names]
    order = [robot.joint_names.index(joint) for joint in joints]
    ctrl_order = []
    model = env.sim.mj_model
    for joint in joints:
        joint_id = model.joint("robot/" + joint).id
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(matches) != 1:
            raise ValueError(f"Expected one actuator for {joint}")
        ctrl_order.append(int(matches[0]))
    wheel = tensor([False, False, False, True] * 2).bool()
    gains = tensor([[60, 60, 92, 0]*2, [1.5, 1.5, 2.3, .5]*2, [60, 60, 60, 12]*2])
    native_targets, probe_actions = [], []
    for vector in np.vstack((np.eye(8), -np.eye(8), np.eye(8) * 6)):
        env.action_manager.process_action(tensor(np.tile(vector, (len(names), 1))))
        env.action_manager.apply_action()
        target = torch.where(wheel, data.joint_vel_target[:, order], data.joint_pos_target[:, order])
        native_targets.append(target[0].cpu().numpy().copy())
        probe_actions.append(vector)
    output_json(out / "mapping_probe.json", dict(actions=np.asarray(probe_actions).tolist(),
                                                native_targets=np.asarray(native_targets).tolist()))
    env.action_manager.process_action(tensor([z["last_action"] for z in pairs]))
    robot.write_joint_state_to_sim(qpos[:, 7:][:, indices], qvel[:, 6:][:, indices])
    env.sim.forward()
    env.sim.sense()
    initial_obs = tensor([z["obs"][0] for z in references])
    vel.wheel_error[:] = initial_obs[:,33] / 5
    vel.heading_integral[:,0] = initial_obs[:,34] / 2
    vel.heading_integral[:,1] = vel.heading_integral[:,0]
    vel.wheel_stationary[:] = tensor([z["position_latched"][0] for z in references]).bool()
    vel.initializing[:] = False
    step = env.sim.step
    policy_tick = 0
    first_delta = {}
    terms = env.termination_manager.active_terms
    events = []

    def record_step():
        q, dq = data.joint_pos[:, order].clone(), data.joint_vel[:, order].clone()
        qt, dqt = data.joint_pos_target[:, order].clone(), data.joint_vel_target[:, order].clone()
        raw = gains[0] * (qt - q) + gains[1] * (dqt - dq)
        ctrl = env.sim.data.ctrl[:, ctrl_order].clone()
        step()
        torques = env.sim.data.qfrc_actuator[:, robot.indexing.joint_v_adr][:, order]
        fields = dict(q_pre=q, dq_pre=dq, q_target=qt, dq_target=dqt, reconstructed_pd=raw,
                      ctrl=ctrl, qfrc_actuator=torques)
        physics.append(policy_tick=policy_tick, physics_tick=len(physics.columns["policy_tick"]),
                       **{key: value.cpu().numpy() for key, value in fields.items()})

    # Compare the first two seconds of constant commands and the sequence through stop re-entry.
    with torch.inference_mode(), patch.object(env.sim, "step", record_step):
        for policy_tick in range(3000):
            ref_idx = [min(policy_tick, len(z["obs"])-1) for z in references]
            cmd = tensor([z["command_applied"][i, :3] for z, i in zip(references, ref_idx)])
            vel.target[:] = cmd
            # The Native command manager advances after integration. Make the command
            # boundary explicit before this observation without advancing its hold state twice.
            vel.command[:] = cmd
            obs = env.observation_manager.compute_group("policy", update_history=False)
            normalized, applied = action(obs)
            all_obs, _, terminated, truncated, _ = env.step(applied)
            rpy = torch.stack(euler_xyz(data.root_link_quat_w), dim=1)
            body = torch.cat((data.root_link_lin_vel_b, data.root_link_ang_vel_b, rpy,
                              (data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2])[:, None]), dim=1)
            policy.append(tick=policy_tick, t_post=(policy_tick+1)*env.step_dt,
                          obs=obs.cpu().numpy(), normalized_action=normalized.cpu().numpy(),
                          action=applied.cpu().numpy(), cmd=cmd.cpu().numpy(), body_gt=body.cpu().numpy(),
                          critic=all_obs["critic"].cpu().numpy(), wheel_latched=vel.wheel_stationary.cpu().numpy(),
                          heading_integral=vel.heading_integral.cpu().numpy())
            if policy_tick == 0:
                first_delta["obs"] = (obs.cpu().numpy() - np.stack([z["obs"][0] for z in references])).tolist()
                np.testing.assert_allclose(obs.cpu().numpy()[:,6:], initial_obs.cpu().numpy()[:,6:],
                                           atol=1e-5, rtol=1e-4)
            if (terminated | truncated).any():
                for i in (terminated | truncated).nonzero().flatten().tolist():
                    reason = [term for term in terms if env.termination_manager.get_term(term)[i]]
                    events.append(dict(case=names[i], tick=policy_tick, reasons=reason))
                break
    for name, trace in (("policy", policy), ("physics", physics)):
        trace.save(out / (name + ".npz"))
    saved_config = json.loads((out.parent / "forward_12/config.json").read_text())
    sys.path.insert(0, str(PACKAGE))
    from sim2sim_ros2_jazzy.adapter import ActionAdapter
    adapter = ActionAdapter(saved_config)
    simulated_targets = np.asarray([adapter.targets(np.asarray(v, dtype=np.float32))
                                    for v in probe_actions_from(out)])
    native_targets = np.asarray(json.loads((out / "mapping_probe.json").read_text())["native_targets"])
    np.testing.assert_allclose(native_targets, simulated_targets, atol=1e-6, rtol=1e-6)
    raw = np.asarray(physics.columns["reconstructed_pd"])
    ctrl = np.asarray(physics.columns["ctrl"])
    actual = np.asarray(physics.columns["qfrc_actuator"])
    limit = gains[2].cpu().numpy()
    clip_error = float(np.max(np.abs(np.clip(raw, -limit, limit)-ctrl)))
    actual_error = float(np.max(np.abs(ctrl-actual)))
    if max(clip_error, actual_error) > 5e-5:
        raise ValueError(f"Native PD contract residuals {clip_error}, {actual_error}")
    first_cases = slice(0, 2)
    native_target = np.where(wheel.cpu().numpy(), np.asarray(physics.columns["dq_target"]),
                             np.asarray(physics.columns["q_target"]))
    def first_pair_delta(field, native):
        sim = np.stack([np.load(out.parent / name / "physics.npz")[field][:4] for name in names[:2]], axis=1)
        return (native[:4, first_cases] - sim).tolist()

    output_json(out / "manifest.json", dict(
        execution_domain="TRAINING_SIM", evidence_status="SUSPICIOUS" if events else "PASS",
        checkpoint_sha256=meta["actor_sha256"], env_sha256=meta["env_config_sha256"],
        source=git_provenance(ORIGINAL), import_root=str(ORIGINAL), cases=names, joints=list(joints),
        dt_policy=env.step_dt, dt_physics=env.physics_dt, ticks=policy_tick+1,
        impratio=env.sim.mj_model.opt.impratio, events=events, initial_obs_delta=first_delta["obs"],
        onnx_same_observation_max_abs=onnx_delta,
        mapping_probe_max_abs=float(np.max(np.abs(native_targets-simulated_targets))),
        reconstructed_clip_max_error_nm=clip_error, ctrl_to_actuator_max_error_nm=actual_error,
        first_four_target_native_minus_sim=first_pair_delta("target", native_target),
        first_four_torque_native_minus_sim=first_pair_delta("qfrc_actuator", actual),
    ))
    print(f"[TRAINING_SIM] Native policy/PD parity, {policy_tick+1} ticks, events={events}", flush=True)


def probe_actions_from(out):
    return json.loads((out / "mapping_probe.json").read_text())["actions"]


def sensor_probe(out):
    """Estimate local policy sensitivity with the same stored state in each pair."""
    import csv
    import torch
    from mj_lab.flash_sac import runtime

    runtime.setup_imports(False)
    from export import load_policy

    ckpt, meta = runtime.read_checkpoint(CKPT, task="flat_roll_pitch_height")
    actor = load_policy(ckpt, meta)
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    frames = []
    for case in ("forward_12", "turn_left", "sequence", "stop"):
        with np.load(out / case / "policy.npz") as trace:
            frames.append(trace["obs"][::15].copy())
    obs = np.concatenate(frames)
    groups = (
        ("gyro", slice(0,3), .25*.2, .25*.05), ("gravity", slice(3,6), .05, .05),
        ("q", slice(9,17), .01, .02), ("dq", slice(17,25), .05*1.5, 0.),
        ("wheel_displacement", slice(33,34), 5*.005, 0.),
        ("heading", slice(34,35), 2*.003, 0.),
    )
    rows = []
    with torch.inference_mode():
        baseline = actor(torch.from_numpy(obs)).numpy()
        for name, indices, half_width, bias in groups:
            draw = rng.uniform(-1., 1., size=obs[:,indices].shape).astype(np.float32)
            samples = [("uniform_0.5x", .5*half_width*draw), ("uniform_1x", half_width*draw)]
            if bias:
                samples += [("bias_plus", bias), ("bias_minus", -bias)]
            for mode, delta in samples:
                changed = obs.copy()
                changed[:,indices] += delta
                # Native position bias is zeroed at the wheel indices before observation noise.
                if name == "q" and mode.startswith("bias_"):
                    changed[:,15:17] = obs[:,15:17]
                error = actor(torch.from_numpy(changed)).numpy()-baseline
                rows.append(dict(group=name, intervention=mode, frames=len(obs),
                                 gain_output_rms=float(np.sqrt(np.mean(error**2))),
                                 wheel_target_raw_rms_rad_s=float(20*np.sqrt(np.mean(error[:,[3,7]]**2))),
                                 gain_output_peak=float(np.abs(error).max())))
    with (out / "sensor_probe.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DESKTOP_DRY_RUN] {len(obs)} saved observations x {len(rows)} perturbations")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("case", choices=(*scenarios(), "native", "sensor_probe"))
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--residual", choices=("on", "off"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-roll", type=float, default=0.)
    parser.add_argument("--clocks", nargs=3, type=int, metavar=("ACTOR", "PD", "PHYSICS"))
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--residual-site-packages", type=Path)
    args = parser.parse_args()
    if args.residual_site_packages is not None:
        sys.path.append(str(args.residual_site_packages))
    if args.case == "native":
        native_run(args.out)
    elif args.case == "sensor_probe":
        sensor_probe(args.out)
    else:
        sim_run(args.out, args.case, args.variant, args.residual, args.seed, args.parity, args.initial_roll, args.clocks)


if __name__ == "__main__":
    main()
