# 고속 전진과 제자리 회전 오차의 원인 검증

실행 도메인 `VALIDATION_SIM` · `TRAINING_SIM` / 브랜치 `agent/codex/straight-heading` / 측정 실행 `PASS` / 기본 sim2sim 속도 선별 `49/58 PASS`

`flat_roll_pitch_height`의 `model_85000`을 고정하고 60초 시험을 146회 수행했다. 좌측 제자리 회전에서 나타난 전진은 sim2sim의 접촉 설정에 크게 민감했다. 고속 전진에서는 두 런타임 모두 정책이 낮은 바퀴 속도 목표를 냈고 실제 속도가 그 목표를 따랐다. 아래 수치는 별도 표기가 없으면 50–60초 평균이다. [전체 조건별 지표](speed_cause/windows.csv) · [좌우 회전 시계열 그림](speed_cause/turn_diagnosis.png)

## 좌측 제자리 회전

| 런타임과 접촉 설정 | 차체 vx (m/s) | 실제 yaw rate (rad/s) | 높이 (m) | 해석 |
| --- | ---: | ---: | ---: | --- |
| sim2sim 기본, impratio 100 | +0.10075 | +1.02384 | 0.2807 | 좌·우 바퀴 목표 −2.935, +4.552 rad/s |
| sim2sim, impratio 1 | +0.00542 | +1.04280 | 0.3196 | 종방향 잔류 속도 94.6% 감소 |
| native mjlab 학습 설정, impratio 1 | −0.00462 | +1.04204 | 0.3192 | 높이와 yaw rate가 두 번째 조건과 근접 |
| native mjlab, impratio 100 | −0.00302 | +1.04236 | 0.3182 | 역방향 개입에서 큰 전진을 재현하지 못함 |

- 기본 sim2sim에서 좌회전 시 바퀴 목표 평균은 +0.808 rad/s, 실제 평균은 +1.003 rad/s였다. 실제 바퀴 축 중앙 속도는 +0.09284 m/s였다. 차체 원점과의 차이 +0.00791 m/s가 전체 전진의 약 8%를 설명한다. 양쪽 바퀴의 평균 접선 슬립 속도는 각각 0.00114, 0.00084 m/s, 평균 수직 하중은 122, 130 N이었다. 마지막 10초 바퀴 토크 절댓값 최댓값은 0.128 N·m 미만이었다. 우회전에서 차체와 축 중앙은 모두 약 +0.0333 m/s였다.
- 좌회전 1초에 yaw rate는 +1.012 rad/s, vx는 −0.054 m/s였다. 15초에는 높이가 0.281 m, 피치가 +3.49°에 이르고 vx가 +0.101 m/s로 정착했다. `impratio`는 접촉의 마찰 방향과 법선 방향의 제약 임피던스 비율에 영향을 주는 솔버 설정이다. 이 개입에서 지면 마찰계수는 계속 0.8이었다. 설정에 따라 정착하는 자세와 정책 출력이 달라졌다. 두 변수 간 단방향 인과관계까지 분리하지는 못했다.
- 두 런타임의 첫 상태를 맞췄을 때 좌회전 actor 관측의 최대 성분 차이는 0.000738이었다. 차이는 IMU 관련 성분에서 시작했다. sim2sim에서 현재 스텝의 자세와 각속도를 읽도록 한 시험에서 관측 차이는 7.5e−9 이내, action 차이는 7.2e−7 이내였다. 첫 물리 스텝 양쪽 바퀴 토크도 1e−5 N·m 이내로 일치했다. 그런데 이 개입만으로는 50–60초 vx가 +0.10075 m/s에 머물렀다. 적분기를 implicitfast로 변경하거나 관절 armature 필드를 0으로 바꾼 개별 시험에서도 같은 좌회전 결과를 얻었다.
- 동일한 기록 관측 18,000개로 PyTorch actor와 ONNX 출력을 비교한 최대 절대차는 3.37e−6이었다. native 역방향 시험 결과를 고려하면, sim2sim의 큰 오차는 시험한 조건에서 `impratio`에 의존하며 다른 런타임 차이와도 상호작용할 수 있다. 하드웨어에서의 접촉 특성은 이번 자료로 판정할 수 없다.

## 명령·높이·초기 피치 개입

| 기본 sim2sim 조건 | 마지막 실제 vx (m/s) | 마지막 실제 yaw rate (rad/s) |
| --- | ---: | ---: |
| 높이 0.335 m, 좌회전 명령 +0.5 / +1.0 / +1.5 rad/s | +0.1502 / +0.1007 / +0.0014 | +0.4851 / +1.0238 / +1.5523 |
| 높이 0.335 m, 우회전 명령 −0.5 / −1.0 / −1.5 rad/s | −0.0076 / +0.0333 / +0.0356 | −0.4706 / −1.0024 / −1.5373 |
| 좌회전 +1.0 rad/s, 높이 명령 0.300 / 0.335 / 0.370 m | +0.0585 / +0.1007 / +0.1269 | +1.0518 / +1.0238 / +1.0027 |

좌회전의 세 높이 명령에서 **실제** 높이는 각각 0.2858 / 0.2807 / 0.2785 m였다. 목표를 높일수록 전진이 늘었지만 실제 높이는 오히려 낮아졌다. 목표 높이가 actor 입력과 자세·바퀴 출력을 함께 바꾸므로 실제 높이만의 효과로 해석하기 어렵다. 공칭 높이와 +1.0 rad/s에서 시작 피치에 −2°, 0°, +2°를 각각 더해도 마지막 vx는 모두 +0.1007 m/s였다. 이 셋은 초기조건 민감도 시험이다.

기존 선별 기준은 전체(0.5–60초)와 마지막(50–60초) 모두에서 vx MAE <0.10 m/s, vy MAE <0.08 m/s, yaw rate MAE <0.10 rad/s다. sim2sim 기본은 49/58, `impratio=1`은 55/58이었다. 기본 설정에서 실패한 9개는 3개 좌회전 조건 × 3개 초기 피치였다. 학습값 적용 후에는 이 9개가 통과했고, +1.4 m/s 전진의 3개 피치 조건에서 vx MAE가 약 0.0894에서 0.1081 m/s로 증가해 기준을 넘었다. 기본 설정 58개 모두 vy와 yaw rate는 기존 선별 기준을 만족했다.

## 고속 직진

기본 sim2sim에서 +0.8 / +1.0 / +1.2 / +1.4 m/s 명령의 실제 vx는 +0.7749 / +0.9478 / +1.1217 / +1.3106 m/s였다. +1.2 m/s에서 반경 0.0925 m로 환산한 순수 구름 목표는 약 12.97 rad/s다. 정책의 바퀴 목표 평균은 12.1258, 실제 평균은 12.1250 rad/s였다. 바퀴 토크 최댓값은 0.034 N·m 미만(한도 12 N·m), 접선 슬립은 약 0.0001 m/s였다. 정상상태에서 바퀴가 받은 낮은 목표 자체가 전진 오차를 거의 전부 설명한다.

native의 +1.2 m/s에서도 바퀴 목표 평균 12.1853 rad/s, 실제 vx +1.1270 m/s였다. 반대로 −1.2 m/s에서 sim2sim과 native는 각각 −1.2193, −1.2260 m/s였다. 정책이 왜 전·후진에서 서로 다른 목표를 선택하도록 학습됐는지는 이번 고정 정책 실험으로 확정할 수 없다. 학습 명령 분포 변경과 episode length·n-step·gamma ablation은 수행하지 않았다. 우선 검증 대상으로 정한 큰 좌회전 전진 출력은 native에서 재현되지 않았다.

## 측정 방법과 재현

- sim2sim 기본 58개, `impratio=1` 58개, IMU·적분기·armature 개별 대조 각 6개. native 기본과 역방향 개입 각 6개. 전진 기준으로 바퀴 부호를 확인했다. 제어 50 Hz로 명령·출력·관절·차체·축 중앙을, 물리 200 Hz로 좌우 제한 전후 토크·하중·슬립을 기록했다. 접선 슬립은 바닥과 접촉하는 바퀴 물질점의 속도를 접선 평면에 투영해 계산했다.
- sim2sim은 동일한 기동 절차와 RL 정지 1초를 거친 후 명령을 인가했다. 피치 개입은 명령 직전 몸체 자세에 가한 상대 회전이다. 접촉 등 개별 설정 변경도 이 시점부터 적용했다. 이전 10개 CSV와 각 3,000스텝의 vx·vy·yaw rate가 모두 정확히 일치했다.
- native는 sim2sim에서 복사한 시작 qpos·qvel·직전 action, 마찰 0.8, 모터 지연 0, 관측 노이즈·랜덤화 해제, 고정 명령과 충분한 시간제한을 사용했다. native의 6개 명령은 제자리 좌·우, ±1.2 m/s 직선, +0.8 m/s와 ±0.8 rad/s의 곡선이었다. 물리 엔진은 각각 CPU MuJoCo와 MuJoCo Warp다.
- 각 시험의 전체·10–20초·50–60초 MAE와 부호 있는 오차 등 438개 행은 CSV에 있다. 원시 NPZ와 실행 로그는 실험 폴더에 로컬 보관된다. 원래 체크포인트의 actor SHA256은 `011c043642752e3e4adc51f8d401d9dbbce8a9d2d98948f813358fde6eb893a5`, ONNX SHA256은 `fe9364a1b7f3b70f60037ba17b76892afdf55ea4561c6d8250c16e9903fd2b68`이다.

실험 worktree 루트에서 다음 명령으로 재현한다. 기존 60초 기준 CSV와 준비된 체크포인트를 사용한다.

```bash
out=docs/exp/2026-11-20/speed_cause_replay
mkdir -p "$out/ros_logs"
source /opt/ros/jazzy/setup.bash
unset ROS_LOCALHOST_ONLY
export OPENBLAS_NUM_THREADS=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_LOG_DIR="$PWD/$out/ros_logs"
for variant in baseline impratio fresh_imu integrator armature; do
  ~/venvs/tita_wbc_jazzy/bin/python test/diagnose_sim2sim_speed.py "$out" --variant "$variant" || exit 1
done
~/venvs/tita_mjlab/bin/python test/diagnose_native_speed.py "$out"
~/venvs/tita_mjlab/bin/python test/diagnose_native_speed.py "$out" --impratio 100
~/venvs/tita_mjlab/bin/python test/summarize_speed_diagnosis.py "$out"
```
