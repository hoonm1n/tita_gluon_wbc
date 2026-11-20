# `model_85000` sim2sim 헤딩 기준선

- 실행 도메인 `VALIDATION_SIM`
- 실행 상태 `PASS`
- 브랜치 `agent/codex/straight-heading` (`9c4ec48` 기준)
- Actor SHA256 `011c043642752e3e4adc51f8d401d9dbbce8a9d2d98948f813358fde6eb893a5`
- ONNX SHA256 `fe9364a1b7f3b70f60037ba17b76892afdf55ea4561c6d8250c16e9903fd2b68`

ROS 2 Jazzy sim2sim에서 저장된 FlashSAC 정책을 MuJoCo 평지, 마찰 0.8, 구형 휠 충돌, 물리 5ms와 정책 20ms 주기로 재생했다. 네 동작을 각각 새 시뮬레이터에서 시작했다. 준비 자세를 거쳐 RL 진입 후 정지 명령을 1초 유지하고, 자세 목표는 롤·피치 0°, 높이 0.335m로 고정했다. 명령을 주기 직전의 위치와 방향을 기준으로 월드 yaw를 연속 각도로 변환하고 시작 직선에 수직인 방향으로 횡이탈을 측정했다.

| 동작 | 구간 | 최종 헤딩 변화 | 최대 절대 헤딩 변화 | 최종 횡이탈 | 최대 절대 횡이탈 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 직진 | 6초 | +4.08° | 4.08° | +0.158m | 0.158m |
| 직진 | 20초 | +14.24° | 14.24° | +1.872m | 1.872m |
| 후진 | 6초 | +2.67° | 2.67° | −0.105m | 0.105m |
| 후진 | 20초 | +5.35° | 5.35° | −0.979m | 0.979m |

20초 직진에서 전후 속도 평균 절대 오차는 0.032m/s, 회전 속도 평균 절대 오차는 0.013rad/s였다. 같은 직진 구간에서 헤딩이 +14.24° 누적됐다. 횡이탈 부호는 시작 자세 기준 좌측이 양수이며, 후진에도 같은 좌표계 부호를 적용했다.

| 동작 | 목표 vx / vy (m/s) | 목표 yaw (rad/s) | 평균 절대 오차 vx / vy (m/s) | 평균 절대 오차 yaw (rad/s) |
| --- | --- | ---: | --- | ---: |
| 직진 20초 | +0.80 / 0 | 0 | 0.032 / 0.0002 | 0.0125 |
| 후진 20초 | −0.80 / 0 | 0 | 0.009 / 0.0002 | 0.0049 |
| 곡선 6초 | +0.80 / 0 | +0.80 | 0.058 / 0.0050 | 0.0147 |
| 제자리 회전 6초 | 0 / 0 | +1.00 | 0.033 / 0.0026 | 0.0336 |

속도와 자세 오차 평균은 명령 직후의 0.5초를 제외했다. 네 시험에서 전체 요청 시간을 채웠으며, 제어 스텝에서 차체·무릎의 바닥 접촉이 관측된 횟수는 0회였다. `PASS`는 해당 조건의 시뮬레이션 재생과 기록 완료를 나타낸다. 성능 수치는 각 명령을 한 번 실행한 명목 평지 기준선이다.

## 검증 증거

- 38개 입력과 8개 출력의 ONNX 모델을 검증했다. 256개 입력에서 PyTorch 대비 최대 절대 출력 오차는 `4.0531e-6`이었다. [ONNX 준비 로그](sim2sim_heading_baseline/prepare.log)
- 정지 명령으로 500제어 스텝을 실행했고 마지막 상태는 `rl`이었다. 네 주행 시험의 CSV 전체 행 수, 측정 종료 시각, 유한값과 체크포인트 해시를 확인했다. [정지 실행 로그](sim2sim_heading_baseline/smoke.log) · [주행 측정 로그](sim2sim_heading_baseline/measurement.log)
- [궤적·헤딩 그래프](sim2sim_heading_baseline/heading_trajectory.png) · [지표 JSON](sim2sim_heading_baseline/result.json) · [직진 CSV](sim2sim_heading_baseline/forward.csv) · [후진 CSV](sim2sim_heading_baseline/reverse.csv) · [곡선 CSV](sim2sim_heading_baseline/curve.csv) · [제자리 회전 CSV](sim2sim_heading_baseline/turn_in_place.csv)

## 재현

실험 작업 폴더 루트에서 검증된 ONNX가 있는 복사본을 사용한다.

```bash
source /opt/ros/jazzy/setup.bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
run_dir="docs/exp/$(date +%F)/sim2sim_heading_replay"
~/venvs/tita_wbc_jazzy/bin/python test/evaluate_sim2sim_heading.py "$run_dir" \
  --checkpoint mj_lab/logs/flash_sac/heading_baseline/model_85000
~/venvs/tita_mjlab/bin/python test/evaluate_sim2sim_heading.py "$run_dir" --plot-only
```
