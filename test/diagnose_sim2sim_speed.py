"""VALIDATION_SIM: fixed-policy wheel, contact and body speed measurements."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from evaluate_sim2sim_heading import LONG_SCENARIOS, REPO, collect_sample

CKPT = REPO / "mj_lab/logs/flash_sac/heading_baseline/model_85000"
BASE = REPO / "docs/exp/2026-11-20/sim2sim_speed_tracking_60s"
NOMINAL = 0.335
SPEED_FIELDS = ["actual_vx_m_s", "actual_vy_m_s", "actual_yaw_rate_rad_s"]
CONTROL = [
    "time_s",
    *SPEED_FIELDS,
    "height_m",
    "pitch_deg",
    "roll_deg",
    "wheel_target_l",
    "wheel_target_r",
    "wheel_speed_l",
    "wheel_speed_r",
    "axle_vx",
    "axle_vy",
    "base_floor_contact",
    "knee_floor_contact",
]
PHYSICS = [
    "time_s",
    "raw_tau_l",
    "raw_tau_r",
    "ctrl_l",
    "ctrl_r",
    "load_l",
    "load_r",
    "slip_l",
    "slip_r",
    "contact_count_l",
    "contact_count_r",
]


def conditions():
    cases = {(c[0], c[2], NOMINAL, 0): n for n, c, _ in LONG_SCENARIOS}
    commands = [(x, 0.0, NOMINAL) for x in (-1.4, -1.2, -1.0, -0.8, 0.8, 1.0, 1.2, 1.4)]
    commands += [(0.0, y, NOMINAL) for y in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)]
    commands += [(0.0, y, h) for y, h in product((-1.0, 1.0), (0.30, 0.37))]
    for (x, y, h), pitch in product(commands, (-2, 0, 2)):
        cases.setdefault((x, y, h, pitch), f"vx{x:+.1f}_yaw{y:+.1f}_h{h:.3f}_p{pitch:+d}")
    return cases


POSE_CASES = {"roll_plus": (10., 0.), "roll_minus": (-10., 0.), "pitch_plus": (0., 5.), "pitch_minus": (0., -5.)}


def tuning_cases():
    cases = [(case, name) for case, name in conditions().items()]
    return cases + [((.8, 0., NOMINAL, 0), name) for name in POSE_CASES]


def native_tuning_names():
    return [name for (x, yaw, h, pitch), name in tuning_cases() if h == NOMINAL and pitch == 0
            and (name in POSE_CASES or (yaw == 0 and abs(x) in (1.2, 1.4)) or yaw != 0)]


def run(job):
    out, (vx, yaw, height, pitch), name, variant, checkpoint = job
    import mujoco as mj
    import rclpy
    from geometry_msgs.msg import Twist, Vector3

    sys.path.insert(0, str(REPO / "sim2sim/sim2sim_ros2_jazzy"))
    from sim2sim_ros2_jazzy.sim2sim_node import Sim2SimNode
    from sim2sim_ros2_jazzy.checkpoint import prepared_config
    from sim2sim_ros2_jazzy.adapter import quat_wxyz_to_rotmat

    config = prepared_config(checkpoint)
    if config is None:
        raise ValueError(f"Missing prepared checkpoint {checkpoint}")
    os.environ["ROS_DOMAIN_ID"] = str(10 + (os.getpid() % 80))
    rclpy.init(args=["--ros-args", "-p", f"config:={config}", "-p", "viewer:=false"])
    node = Sim2SimNode()
    env, adapter = node.env, node.action_adapter
    m, d = env.model, env.data
    wheels = np.array([3, 7])
    bodies = [m.body(side + "_leg_4").id for side in ("left", "right")]
    geoms = [
        next(i for i in range(m.ngeom) if m.geom_bodyid[i] == body and m.geom_contype[i])
        for body in bodies
    ]
    floor = m.geom("floor").id
    jac = np.empty((3, m.nv))
    force = np.empty(6)
    physics, control, observations, actions, joints = [], [], [], [], []
    try:
        for _ in range(250):
            node._step()
            if node.sm.state == "rl":
                break
        else:
            raise RuntimeError("Did not enter RL")
        for _ in range(50):
            node._step()
        if pitch:
            delta = np.deg2rad(pitch / 2)
            q = d.qpos[3:7].copy()
            mj.mju_mulQuat(d.qpos[3:7], q, np.array([np.cos(delta), 0.0, np.sin(delta), 0.0]))
            mj.mj_forward(m, d)
        initial = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), last_action=node.last_action.copy())
        # Switch one solver/observation setting only after the shared preparation.
        if variant == "impratio":
            m.opt.impratio = 1.0
        elif variant == "integrator":
            m.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
        elif variant == "armature":
            m.dof_armature[env.qvel_addr] = 0.0
        elif variant == "fresh_imu":
            env.base_quat_sensor = env.base_quat
            env.base_ang_vel = lambda: d.qvel[3:6].astype(np.float32).copy()
        roll_goal, pitch_goal = POSE_CASES.get(name, (0., 0.))
        node._cmd_pose_cb(Vector3(x=np.deg2rad(roll_goal), y=np.deg2rad(pitch_goal), z=height))
        msg = Twist()
        msg.linear.x, msg.angular.z = vx, yaw
        node._cmd_twist_cb(msg)
        # Wheel axes must both give positive rolling velocity along body +x.
        body_x = quat_wxyz_to_rotmat(env.base_quat())[:, 0]
        for body in bodies:
            axis = d.xaxis[m.body_jntadr[body]]
            assert np.dot(np.cross(axis, [0, 0, 1]), body_x) > 0.8
        original_step, original_infer = mj.mj_step, node.policy.run

        def infer(outputs, inputs):
            result = original_infer(outputs, inputs)
            observations.append(inputs["nn_input0"].ravel().copy())
            actions.append(result[0].ravel().copy())
            return result

        def step(model, data):
            pre_vel = d.qvel.copy()
            original_step(model, data)
            # mj_step leaves Cartesian kinematics/contacts at the pre-integration state.
            load = np.zeros(2)
            slips = np.zeros(2)
            counts = np.zeros(2)
            for i, contact in enumerate(d.contact):
                if floor not in contact.geom or contact.efc_address < 0:
                    continue
                other = contact.geom[1] if contact.geom[0] == floor else contact.geom[0]
                if other not in geoms:
                    continue
                k = geoms.index(other)
                mj.mj_contactForce(m, d, i, force)
                mj.mj_jac(m, d, jac, None, contact.pos, bodies[k])
                tangent = contact.frame.reshape(3, 3)[1:] @ (jac @ pre_vel)
                load[k] += force[0]
                slips[k] += np.linalg.norm(tangent)
                counts[k] += 1
            target = adapter.targets(actions[-1])[wheels]
            raw = adapter.kd[wheels] * (target - pre_vel[env.qvel_addr[wheels]])
            physics.append(
                [
                    (len(physics) + 1) * m.opt.timestep,
                    *raw,
                    *d.ctrl[np.array(env.actuator_ids)[wheels]],
                    *load,
                    *(slips / np.maximum(counts, 1)),
                    *counts,
                ]
            )

        # Compute axle-center velocities at the post-step pose using a scratch MjData.
        scratch = mj.MjData(m)
        with patch.object(mj, "mj_step", step), patch.object(node.policy, "run", infer):
            for tick in range(1, 3001):
                node._step()
                if node.sm.state != "rl":
                    raise RuntimeError("Left RL")
                sample = collect_sample(node, tick * 0.02, node.command)
                scratch.qpos[:] = d.qpos
                mj.mj_kinematics(m, scratch)
                mj.mj_comPos(m, scratch)
                centers = []
                for body, geom in zip(bodies, geoms):
                    mj.mj_jac(m, scratch, jac, None, scratch.geom_xpos[geom], body)
                    centers.append(jac @ d.qvel)
                axle = quat_wxyz_to_rotmat(env.base_quat()).T @ np.mean(centers, axis=0)
                control.append(
                    [
                        tick * 0.02,
                        *(sample[k] for k in SPEED_FIELDS),
                        sample["height_m"],
                        sample["pitch_deg"],
                        sample["roll_deg"],
                        *adapter.targets(actions[-1])[wheels],
                        *env.joint_vel()[wheels],
                        *axle[:2],
                        sample["base_floor_contact"],
                        sample["knee_floor_contact"],
                    ]
                )
                joints.append(env.joint_pos())
        arrays = dict(control=control, physics=physics, obs=observations, actions=actions, joints=joints)
        if not all(np.isfinite(arr).all() for arr in arrays.values()):
            raise ValueError("Nonfinite telemetry")
        assert np.allclose(
            np.asarray(physics)[:, 3:5], np.clip(np.asarray(physics)[:, 1:3], -12, 12), atol=2e-6
        )
        if checkpoint.resolve() == CKPT.resolve() and variant == "baseline" and (BASE / (name + ".csv")).exists():
            ref = np.genfromtxt(BASE / (name + ".csv"), names=True, delimiter=",")[1:]
            for j, key in enumerate(SPEED_FIELDS):
                np.testing.assert_array_equal(ref[key], np.asarray(control)[:, j + 1])
        np.savez_compressed(
            out / (name + ".npz"),
            **arrays,
            **initial,
            condition=[vx, yaw, height, pitch],
            pose_goal_deg=[roll_goal, pitch_goal],
            control_fields=CONTROL,
            physics_fields=PHYSICS,
        )
        return name
    finally:
        env.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    cases_group = parser.add_mutually_exclusive_group()
    cases_group.add_argument("--case")
    cases_group.add_argument("--cases", nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument(
        "--variant",
        choices=("baseline", "impratio", "integrator", "armature", "fresh_imu"),
        default="baseline",
    )
    args = parser.parse_args()
    if args.variant != "baseline":
        args.out = args.out / args.variant
    args.out.mkdir(parents=True, exist_ok=True)
    cases = tuning_cases() if args.finetune else conditions().items()
    jobs = [
        (args.out, case, name, args.variant, args.checkpoint)
        for case, name in cases
        if (not args.case or name == args.case) and (not args.cases or name in args.cases)
        and (
            args.variant in ("baseline", "impratio")
            or name
            in {"turn_left", "turn_right", "forward_12", "reverse_12", "curve_left", "curve_right"}
        )
    ]
    if not jobs:
        parser.error("Unknown case")
    with ProcessPoolExecutor(args.workers) as pool:
        for name in pool.map(run, jobs):
            print(f"[VALIDATION_SIM] PASS {name}", flush=True)
