# `model_85000` sim2sim 60초 선속도·Yaw Rate 추종

- 실행 도메인 `VALIDATION_SIM`
- 실행 상태 `PASS` · 속도 추종 선별 기준 9/10 통과
- 브랜치 `agent/codex/straight-heading`
- Actor SHA256 `011c043642752e3e4adc51f8d401d9dbbce8a9d2d98948f813358fde6eb893a5`
- ONNX SHA256 `fe9364a1b7f3b70f60037ba17b76892afdf55ea4561c6d8250c16e9903fd2b68`

`model_85000`으로 평지 sim2sim에서 명령 10종을 각각 60초 유지했다. 매 시험은 새 시뮬레이터에서 RL 진입 후 정지 명령을 1초 유지한 뒤 시작했다. 지면 마찰 0.8, 물리 5ms와 제어 20ms, 목표 롤·피치 0°와 높이 0.335m를 적용했다. `vx`, `vy`, yaw rate는 몸체 기준이며 모든 `vy` 목표는 0이다. 각 명령을 한 번 실행했다.

## 속도 추종

0.5–60초와 50–60초 평균 절대 오차에 `vx` < 0.10m/s, `vy` < 0.08m/s, yaw rate < 0.10rad/s를 적용했다. 10개 모두 60초를 완료했고, 제어 스텝에서 관측된 차체·무릎의 바닥 접촉은 0회였다. `vy`와 yaw rate는 양쪽 구간 모두 10/10 통과했다. [기존 시뮬레이션 선별 기준](../2026-10-29/pose_policy_selection_summary.md#protocol)

| 명령 `vx`, yaw rate | `vx` MAE 전체 / 마지막 10초 (m/s) | `vy` MAE 마지막 10초 (m/s) | Yaw rate MAE 전체 / 마지막 10초 (rad/s) | 판정 |
| --- | ---: | ---: | ---: | --- |
| `+0.4, 0` | 0.0259 / 0.0267 | <0.001 | 0.0194 / 0.0094 | PASS |
| `+0.8, 0` | 0.0275 / 0.0251 | <0.001 | 0.0137 / 0.0152 | PASS |
| `+1.2, 0` | 0.0828 / 0.0783 | <0.001 | 0.0045 / 0.0053 | PASS |
| `−0.4, 0` | 0.0281 / 0.0315 | <0.001 | 0.0155 / 0.0088 | PASS |
| `−0.8, 0` | 0.0081 / 0.0081 | <0.001 | 0.0033 / 0.0035 | PASS |
| `−1.2, 0` | 0.0189 / 0.0193 | <0.001 | 0.0102 / 0.0120 | PASS |
| `+0.8, +0.8` | 0.0263 / 0.0220 | 0.0050 | 0.0465 / 0.0502 | PASS |
| `+0.8, −0.8` | 0.0418 / 0.0414 | 0.0049 | 0.0067 / 0.0060 | PASS |
| `0, +1.0` | 0.0911 / **0.10075** | 0.0048 | 0.0262 / 0.0238 | FAIL (`vx`) |
| `0, −1.0` | 0.0327 / 0.0333 | 0.0016 | 0.0024 / 0.0024 | PASS |

- 직진 +1.2m/s는 마지막 10초에 실제 평균 +1.1217m/s로 약 6.5% 느렸다. `vx` MAE는 10–20초 0.0803m/s, 50–60초 0.0783m/s로 일정한 과소 추종이었다.
- 좌측 곡선의 마지막 10초 실제 평균 yaw rate는 +0.8502rad/s였다. 10–20초의 yaw rate MAE 0.05013rad/s와 50–60초의 0.05015rad/s는 비슷했다. 우측 곡선의 마지막 10초 실제 평균은 −0.8060rad/s였다.
- 좌측 제자리 회전에서 `vx` 목표는 0이었고 실제 평균은 10–20초 +0.09765m/s, 50–60초 +0.10075m/s였다. 같은 마지막 구간의 실제 평균 yaw rate는 +1.0238rad/s였다. 우측 회전에서 마지막 10초의 평균 `vx`는 +0.03331m/s였다.

## 직선 기준 헤딩과 횡이탈

명령 시작 시점의 방향과 위치를 기준으로 계산했다. 양의 횡이탈은 시작 자세 기준 왼쪽이다.

| `vx` 명령 (m/s) | 20초 헤딩 변화 / 횡이탈 | 60초 헤딩 변화 / 횡이탈 |
| ---: | ---: | ---: |
| +0.4 | +32.91° / +2.23m | +67.28° / +14.03m |
| +0.8 | +14.24° / +1.87m | +46.95° / +17.22m |
| +1.2 | +3.52° / +0.55m | +14.49° / +7.33m |
| −0.4 | +26.37° / −2.08m | +53.56° / −13.42m |
| −0.8 | +5.35° / −0.98m | −0.25° / −2.82m |
| −1.2 | −9.90° / +2.10m | −36.40° / +20.88m |

## 재현 및 데이터

[직진·후진 속도 그래프](sim2sim_speed_tracking_60s/speed_tracking_straight.png) · [곡선·제자리 회전 속도 그래프](sim2sim_speed_tracking_60s/speed_tracking_turning.png) · [궤적·헤딩 그래프](sim2sim_speed_tracking_60s/heading_trajectory.png) · [지표 JSON](sim2sim_speed_tracking_60s/result.json) · [시험별 CSV와 로그](sim2sim_speed_tracking_60s/)

샘플 30,010개의 유한값과 해시를 확인했다. 이전 직진·후진·좌측 곡선·좌측 제자리 회전 시험과 같은 길이까지의 CSV 데이터가 정확히 일치했다.

```bash
source /opt/ros/jazzy/setup.bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
run_dir="docs/exp/$(date +%F)/sim2sim_speed_tracking_60s_replay"
export ROS_LOG_DIR="$PWD/$run_dir/ros_logs"
~/venvs/tita_wbc_jazzy/bin/python test/evaluate_sim2sim_heading.py "$run_dir" \
  --checkpoint mj_lab/logs/flash_sac/heading_baseline/model_85000 --long-duration
~/venvs/tita_mjlab/bin/python test/evaluate_sim2sim_heading.py "$run_dir" --plot-only
```
