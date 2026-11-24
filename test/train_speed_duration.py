"""TRAINING_SIM: run the speed-command duration ablation through the normal trainer."""

import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mj_lab.robots.tita.common import mdp
from mj_lab.scripts import train


class LongTurnCommand(mdp.VelocityCommand):
    """Preserve B's command ranges and match its turn exposure via draw probability."""

    draw_probability = 0.055

    def _resample_command(self, env_ids):
        target = torch.rand((len(env_ids), 3), device=self.device)
        bounds = (self.cfg.lin_vel_x, self.cfg.lin_vel_y, self.cfg.ang_vel_z)
        for axis, (low, high) in enumerate(bounds):
            target[:, axis] = low + target[:, axis] * (high - low)
        probability = self.draw_probability
        draw = torch.rand(len(env_ids), device=self.device)
        is_turn = draw < probability
        # Rescale the non-turn portion to the original 0.85 probability mass.
        non_turn = (draw - probability) * (0.85 / (1 - probability))
        is_stop = ~is_turn & (non_turn < 0.10)
        is_fast = (non_turn >= 0.10) & (non_turn < 0.25)
        x, y, yaw = target.unbind(dim=1)
        fast = torch.copysign(1.0 + x.abs() / 3, x)
        turn = torch.copysign(0.4 + 0.4 * yaw.abs(), yaw)
        zero = torch.zeros_like(x)
        self.target[env_ids] = torch.stack((
            torch.where(is_turn | is_stop, zero, torch.where(is_fast, fast, x)),
            y,
            torch.where(is_turn, turn, torch.where(is_stop | is_fast, zero, yaw)),
        ), dim=1)
        duration = 20.0 + 10.0 * torch.rand(len(env_ids), device=self.device)
        self.time_left[env_ids] = torch.where(is_turn, duration, self.time_left[env_ids])


class ExposureLoggingEnv(train.TitaFlashSACEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command = mdp.velocity(self.env)
        self.previous_yaw = torch.zeros(self.num_envs, device=self.device)
        self.turn_age = torch.zeros_like(self.previous_yaw)
        self.turn_ticks = torch.zeros((), device=self.device, dtype=torch.int64)
        self.long_turn_ticks = torch.zeros_like(self.turn_ticks)
        self.ticks = 0

    def step(self, actions):
        x, y, yaw = self.command.command.unbind(dim=1)
        turning = (x == 0) & (y == 0) & (yaw != 0)
        continued = turning & (yaw == self.previous_yaw)
        self.turn_age = torch.where(continued, self.turn_age + 1, turning.float())
        self.previous_yaw.copy_(yaw)
        self.turn_ticks += turning.sum()
        self.long_turn_ticks += (self.turn_age * self.env.step_dt >= 15).sum()
        self.ticks += self.num_envs
        next_obs, replay_next, reward, terminated, truncated, info = super().step(actions)
        self.previous_yaw.masked_fill_(terminated | truncated, 0)
        info["log"]["pure_turn_step_fraction"] = self.turn_ticks / self.ticks
        info["log"]["pure_turn_at_least_15s_step_fraction"] = self.long_turn_ticks / self.ticks
        return next_obs, replay_next, reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("E", "H"))
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    steps = os.environ.get("TITA_DURATION_STEPS", "10000")
    env_cfg = train.TitaFlatRollPitchHeightEnvCfg

    def config(num_envs):
        cfg = env_cfg(num_envs)
        cfg.episode_length_s = 60.0
        return cfg

    sys.argv[1:] = [
        "--task", "flat_roll_pitch_height", "--headless", "--speed-ablation", "B",
        "--warm_start", "mj_lab/logs/flash_sac/heading_baseline/model_85000",
        "--num_envs", "4096", "--steps", steps, "--seed", str(args.seed),
        "--run_name", f"speed_duration_{args.arm}_{args.seed}" + ("_smoke" if steps != "10000" else ""),
        "--save_every", "5000", "--log_every", "100", "--video_interval", "0", "--logger", "console",
    ]
    print(f"[TRAINING_SIM] {args.arm=} {args.seed=} episode=60s "
          f"turn_hold={'20–30s' if args.arm == 'H' else '6–8s'} "
          f"turn_draw={LongTurnCommand.draw_probability if args.arm == 'H' else 0.15}", flush=True)
    command = LongTurnCommand if args.arm == "H" else mdp.VelocityCommand
    with (
        patch.object(train, "TitaFlatRollPitchHeightEnvCfg", config),
        patch.object(train, "TitaFlashSACEnv", ExposureLoggingEnv),
        patch.object(mdp, "VelocityCommand", command),
    ):
        train.main()


if __name__ == "__main__":
    main()
