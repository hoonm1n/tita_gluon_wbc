"""TRAINING_SIM: native mjlab replay initialized from the paired ROS/MuJoCo trace."""

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

from diagnose_sim2sim_speed import CKPT, REPO, native_tuning_names

sys.path.insert(0, str(REPO))
from mj_lab.flash_sac import runtime
from mj_lab.robots.tita.common.robot_cfg import JOINTS

NAMES = ("turn_left", "turn_right", "forward_12", "reverse_12", "curve_left", "curve_right")


@torch.inference_mode()
def measure(out, impratio, checkpoint=CKPT, finetune=False):
    runtime.setup_imports(False)
    from export import load_policy
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, euler_xyz_from_quat

    ckpt, meta = runtime.read_checkpoint(checkpoint, task="flat_roll_pitch_height")
    policy = load_policy(ckpt, meta).cuda()

    def act(obs):
        normalized = policy.actor.get_mean_and_std(obs, training=False)[0].tanh()
        return normalized * torch.where(
            normalized >= 0, policy.action_gain_positive, policy.action_gain_negative
        )

    names = native_tuning_names() if finetune else NAMES
    traces = [np.load(out / (name + ".npz")) for name in names]

    def batch(key):
        return torch.as_tensor(np.stack([z[key] for z in traces]), device="cuda", dtype=torch.float32)

    parity = []
    for z in traces:
        predicted = act(torch.as_tensor(z["obs"], device="cuda"))
        expected = torch.as_tensor(z["actions"], device="cuda")
        torch.testing.assert_close(predicted, expected, atol=1e-5, rtol=1e-4)
        parity.append((predicted - expected).abs().max().item())

    with (ckpt / "env.pkl").open("rb") as stream:
        cfg = pickle.load(stream)
    cfg.seed, cfg.scene.num_envs, cfg.episode_length_s = 42, len(traces), 65.0
    friction = cfg.events["physics_material"]
    friction.params["ranges"] = (0.8, 0.8)
    cfg.events = {"physics_material": friction}
    for motor in cfg.scene.entities["robot"].articulation.actuators:
        motor.delay_min_lag = motor.delay_max_lag = 0
    cfg.observations["policy"].enable_corruption = False
    cfg.commands["base_velocity"].sensor_bias = False
    if impratio is not None:
        cfg.sim.mujoco.impratio = impratio
    if finetune:
        # Observe contacts for the full 60 s instead of resetting and hiding the failure.
        cfg.terminations.pop("base_contact")
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        env.reset()
        robot = env.scene["robot"]
        vel = env.command_manager.get_term("base_velocity")
        pose = env.command_manager.get_term("base_pose")
        pose.target[:] = pose.command[:] = pose.command.new_tensor([0, 0, 0.335])
        if finetune:
            pose.target[:, :2] = torch.deg2rad(batch("pose_goal_deg"))
        params = batch("condition")
        vel.command[:] = vel.target[:] = torch.stack(
            [params[:, 0], params[:, 0] * 0, params[:, 1]], dim=-1
        )
        vel.time_left[:] = pose.time_left[:] = 1e9
        vel.gyro_bias.zero_()
        vel.radius_scale.fill_(1.0)
        qpos, qvel = batch("qpos"), batch("qvel")
        root = qpos[:, :7].clone()
        root[:, :3] += env.scene.env_origins
        robot.write_root_link_pose_to_sim(root)
        robot.write_root_link_velocity_to_sim(
            torch.cat((qvel[:, :3], quat_apply(root[:, 3:], qvel[:, 3:6])), dim=1)
        )
        indices = [JOINTS.index(joint) for joint in robot.joint_names]
        robot.write_joint_state_to_sim(qpos[:, 7:][:, indices], qvel[:, 6:][:, indices])
        env.action_manager.process_action(batch("last_action"))
        env.sim.forward()
        obs = env.observation_manager.compute(update_history=True)["policy"]
        first = obs.cpu().numpy()
        reference = np.stack([z["obs"][0] for z in traces])
        np.testing.assert_allclose(first[:, 6:], reference[:, 6:], atol=1e-5)
        sim_step = env.sim.step
        first_tick_torques = []

        def trace_first_tick():
            sim_step()
            torque = env.sim.data.qfrc_actuator[:, robot.indexing.joint_v_adr]
            first_tick_torques.append(
                torque[:, [robot.joint_names.index(JOINTS[i]) for i in (3, 7)]].cpu().numpy().copy()
            )

        env.sim.step = trace_first_tick
        rows, obs_history, act_history = [], [], []
        wheel_ids = [robot.body_names.index(side + "_leg_4") for side in ("left", "right")]
        contact_rows, axle_rows = [], []
        for _ in range(3000):
            action = act(obs)
            obs_history.append(obs.cpu().numpy().copy())
            act_history.append(action.cpu().numpy().copy())
            all_obs, _, term, trunc, _ = env.step(action)
            if len(rows) == 0:
                env.sim.step = sim_step
            if (term | trunc).any():
                raise ValueError("Native episode terminated before measurement end")
            obs = all_obs["policy"]
            d = robot.data
            roll, pitch, _ = euler_xyz_from_quat(d.root_link_quat_w)
            measured = torch.stack(
                (
                    d.root_link_lin_vel_b[:, 0],
                    d.root_link_lin_vel_b[:, 1],
                    d.root_link_ang_vel_b[:, 2],
                    d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2],
                    torch.rad2deg(pitch),
                    torch.rad2deg(roll),
                    (action[:, 3] * 20).clamp(-100, 100),
                    (action[:, 7] * 20).clamp(-100, 100),
                    d.joint_vel[:, robot.joint_names.index(JOINTS[3])],
                    d.joint_vel[:, robot.joint_names.index(JOINTS[7])],
                ),
                dim=1,
            )
            if finetune:
                contact = torch.stack([(env.scene[sensor].data.found > 0).any(dim=1)
                                      for sensor in ("base", "knees")], dim=1)
                contact_rows.append(contact.cpu().numpy())
                axle = quat_apply_inverse(d.root_link_quat_w, d.body_link_lin_vel_w[:,wheel_ids].mean(dim=1))
                axle_rows.append(axle[:,:2].cpu().numpy())
            rows.append(measured.cpu().numpy())
        values = np.asarray(rows)
        assert np.isfinite(values).all()
        stem = "native" if impratio is None else f"native_impratio{impratio:g}"
        np.savez_compressed(
            out / (stem + ".npz"),
            first_torques=first_tick_torques,
            names=names,
            values=values,
            obs=obs_history,
            actions=act_history,
            fields=traces[0]["control_fields"][1:11],
            contacts=contact_rows, axle=axle_rows,
        )
        diagnostic = dict(
            names=names,
            same_input_max_abs_action_error=parity,
            initial_observation_delta=(first - np.stack([z["obs"][0] for z in traces])).tolist(),
            sha256=meta["actor_sha256"],
            effective_impratio=env.sim.mj_model.opt.impratio,
            integrator=int(env.sim.mj_model.opt.integrator),
            dof_armature=env.sim.mj_model.dof_armature.tolist(),
        )
        (out / (stem + "_parity.json")).write_text(json.dumps(diagnostic, indent=2) + "\n")
        print(f"[TRAINING_SIM] PASS {len(names)} x 60 s; action parity", max(parity))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--impratio", type=float)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()
    measure(args.out, args.impratio, args.checkpoint, args.finetune)
