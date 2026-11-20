# TITA sim2sim

Execution Domain: VALIDATION_SIM

Evidence Status: PASS

Policy Checkpoint SHA256: 011c043642752e3e4adc51f8d401d9dbbce8a9d2d98948f813358fde6eb893a5

ONNX SHA256: fe9364a1b7f3b70f60037ba17b76892afdf55ea4561c6d8250c16e9903fd2b68

Observed metrics / behavior / log summary:

```json
{
  "execution_domain": "VALIDATION_SIM",
  "evidence_status": "PASS",
  "checkpoint": "/home/kim/tita_ws/tita_gluon_wbc_heading/mj_lab/logs/flash_sac/heading_baseline/model_85000/actor.pt",
  "checkpoint_sha256": "011c043642752e3e4adc51f8d401d9dbbce8a9d2d98948f813358fde6eb893a5",
  "onnx_sha256": "fe9364a1b7f3b70f60037ba17b76892afdf55ea4561c6d8250c16e9903fd2b68",
  "policy_steps": 500,
  "headless": true,
  "residual_wheel_pd": false,
  "fixed_residual_lqr": false,
  "velocity_bias": false,
  "leg_action_filter": "none",
  "leg_notch_hz": 8.3,
  "leg_notch_q": 3.0,
  "support_mode": null,
  "command_limits": {
    "lin_vel_x": [
      -1.5,
      1.5
    ],
    "lin_vel_y": [
      0.0,
      0.0
    ],
    "ang_vel_z": [
      -3.0,
      3.0
    ]
  },
  "pose_command_config": {
    "roll_range": [
      -0.5235987755982988,
      0.5235987755982988
    ],
    "pitch_range": [
      -0.2617993877991494,
      0.2617993877991494
    ],
    "height_range": [
      0.27,
      0.42
    ],
    "nominal_height": 0.335,
    "ramp_rates": [
      0.15,
      0.15,
      0.02
    ]
  },
  "fixed_command": [
    0.0,
    0.0,
    0.0
  ],
  "detail": "Finite MuJoCo playback completed; this is not a locomotion quality evaluation.",
  "tilt_faults": 0,
  "final_state": "rl",
  "pose_target_rad_m": [
    0.0,
    0.0,
    0.33500000834465027
  ],
  "pose_command_rad_m": [
    0.0,
    0.0,
    0.33500000834465027
  ],
  "pose_actual_rad_m": [
    -0.002495378255844116,
    -0.0485420860350132,
    0.36082151532173157
  ],
  "final_base_position_m": [
    -0.28916099667549133,
    0.019371015951037407,
    0.36082151532173157
  ],
  "viewer_threads_stopped": true
}
```
