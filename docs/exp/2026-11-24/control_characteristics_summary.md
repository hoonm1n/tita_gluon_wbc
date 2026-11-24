# TITA 고정 정책 제어 성능과 원인 분석

- 실행 도메인 `TRAINING_SIM`, `VALIDATION_SIM`, `DESKTOP_DRY_RUN`
- 브랜치 `agent/codex/straight-heading`, 기준 commit `7a37f49`
- 85000 actor SHA-256 `011c043642752e3e4adc51f8d401d9dbbce8a9d2d98948f813358fde6eb893a5`
- 85000 ONNX SHA-256 `fe9364a1b7f3b70f60037ba17b76892afdf55ea4561c6d8250c16e9903fd2b68`
- Teacher80k actor SHA-256 `5487e3584fddbdfc87b7485bb2ad4c7da8069a7679b1975789d64419e7482334`
- Residual10k actor SHA-256 `d6672529ee247a92b7043f50bb6deb501a65becbf760761a412b33d5701b6002`
- 증거 상태 `PASS`는 계측 검증과 요청 구간 완료를 뜻한다. 저마찰 전진 4회는 base 접촉의 `SUSPICIOUS` 사건으로 기록했다. 실기 성능은 `NOT_TESTED`다.

**고속 전진 과소 추종은 actor의 바퀴 목표에서 형성됐다. 좌측 제자리 회전은 첫 물리 틱부터 접촉력이 달라졌고, 그 차이가 다음 정책 틱의 바퀴·다리 목표까지 전달됐다. 공칭 정지에서도 속도가 거의 0이 되기 전 약 0.69 m 밀렸으므로, 마지막 10초 속도 오차만으로 위치 유지 성능을 대표할 수 없다.**

최종 집계는 Sim2Sim 112회와 Native 3조건이다. Sim2Sim에는 Teacher80k ON/OFF 8회가 포함된다. MuJoCo 3.10.0/Jazzy에서 85000 기준을 재생했고 Native는 원본 학습 소스 해시 36개를 검사한 뒤 mjlab/MuJoCo Warp 3.11.0에서 실행했다. Residual 두 팔은 같은 3.11.0/PyTorch 경로와 50/250/500 Hz 구성을 사용했다. 원본 정책은 모두 고정했다. [실행 목록](control_characteristics/run_index.csv) · [시스템 계약](tita_control_system_reference.md)

## E01–E07, E15 기본 성능과 오차 위치

- **E01 일관성.** 새 계측을 붙인 60초 고속 전진·좌회전의 몸체 속도 시계열은 기존 실행기 및 보관 CSV와 값 단위로 일치했다. 동일 관측의 PyTorch–ONNX 최대 action 차는 `2.76e−6`이었다. 24개 관절별 단일 입력의 두 target 매핑 최대차는 0이었다. Native의 재구성 PD clip, motor ctrl, 일반화 구동 토크는 8축 모두 일치했다. Native와 기본 Sim2Sim의 첫 관측 차 최댓값은 gyro·중력 채널에서 `7.38e−4`였고, 고속 전진과 좌회전 첫 토크 차 최댓값은 각각 `0.027`, `0.016 N·m`였다. 고속 전진은 두 번째 물리 틱부터 차이가 커져 네 번째 최대차가 `3.16 N·m`였다. 이 시점에는 엔진별 관절 피드백도 달라졌다. 기존의 fresh IMU 개입 자료는 IMU 시각 차로 첫 관측이 달라진다는 보조 증거다. [Native 계측](control_characteristics/native/manifest.json) · [기존 fresh IMU 대조](../2026-11-20/speed_tracking_cause_summary.md)
- **E02 과도.** `+0.8 m/s` 전진 10–90% rise time은 step `1.26 s`, 1초 ramp `1.70 s`였고 분석용 정착시간은 `1.84→2.68 s`였다. `+1 rad/s` 좌회전의 yaw rise time은 `0.04→0.82 s`였다. 두 속도축의 20초 평균 상태는 step과 ramp에서 가까웠다. roll·pitch·height 요청 step은 각각 저장된 변화율 제한을 거친 입력으로 평가했다. [과도응답 그림](control_characteristics/steps_and_ramps.png) · [창별 지표](control_characteristics/metrics.csv)
- **E03 정상상태.** 정지 명령 `60 s`의 마지막 10초 `vx` MAE는 약 `0.000006 m/s`였지만 시작점에서 `0.694 m` 떨어졌다. 120초 재생에서도 `0.693 m`였다. 2.10–16.06초 사이 정지 위치 latch가 합계 24틱 풀렸고 바퀴 변위 관측 최대치는 `0.093 m`였다. 관측이 world 위치 오차를 그대로 나타내지 않는다는 것을 확인했다. 속력 gate `0.10→0.08/0.12 m/s`만 바꾼 진단에서는 60초 변위가 `0.980/0.892 m`로 늘었다. 정지 위치 유지는 gate 값 하나로 설명하기 어렵다. [정지 응답](control_characteristics/stop_hold.png) · [변위 지표](control_characteristics/transitions.csv)
- **E04–E05 하위 제어와 포화.** `+1.2 m/s`의 마지막 10초 실제 몸체 `vx`는 Sim2Sim `1.1217`, Native `1.1270 m/s`였다. Sim2Sim 바퀴 평균 target/GT는 `12.1258/12.1250 rad/s`였고 순수 구름 기준은 `12.97 rad/s`다. 바퀴 Kd `0.8/1.2×`를 적용해도 `vx` 오차는 약 `0.080/0.078 m/s`였다. 이 운전점의 정상상태 부족분은 낮은 target과 일치한다. `+1.2` step에는 바퀴 토크 clip이 `5 ms` 있었고, 연속 전환 및 yaw 극단 시험에서는 바퀴 clip이 최대 `20/40 ms`였다. 시험한 극단 명령에서 joint target clip은 관측되지 않았다. [토크·target 집계](control_characteristics/joint_metrics.csv) · [바퀴·접촉 집계](control_characteristics/contact_metrics.csv)
- **E06 heading.** 기존 직선 궤적의 시작 자세 기준 60초 횡이탈은 `+0.8`에서 `+17.22 m`였다. 그중 전진속도 방향의 회전 성분이 `+17.218 m`, 몸체 횡속도 성분은 `−0.0024 m`였다. `−0.8`은 최종 heading `−0.25°`였지만 최대 편차 `5.37°`, 최종 횡이탈 `−2.82 m`였다. 기존 CSV에는 z 속도가 없어 나머지 성분은 적분 잔차로 남겼다. [직진·후진 분해](control_characteristics/heading_archive.csv)
- **E07 운동학.** 구형 바퀴 접점의 관절 Jacobian으로 body 회전과 다리 속도를 반영했다. 두 바퀴 접촉으로 제한했을 때 `r·평균(dq)` 대비 FK의 `vx` RMSE는 고속 전진 전체 `0.0996→0.00305`, 좌회전 전체 `0.00865→0.00055`, 연속 전환 `0.155→0.00620 m/s`였다. FK의 입력은 시뮬레이션 자세와 encoder·gyro에 해당하는 값이다. 실제 접촉점 slip과 기하학적으로 가정한 무미끄럼 추정값을 따로 집계했다. [운동학 오차](control_characteristics/odometry.csv) · [접촉·slip](control_characteristics/contact_metrics.csv)
- **E15 전환.** 전진→후진과 후진→전진의 속도 영점 통과는 각각 `0.32/0.34 s`, 그때까지의 이동량은 `0.153/0.143 m`였다. 곡선→양의 제자리 회전에서 yaw rate overshoot는 약 `40%`였고 단독 좌회전 step은 약 `5.7%`였다. 마지막 정지의 위치 latch는 양 방향 시퀀스 모두 `0.08 s`에 다시 활성화됐다. 출발 전 이력과 무관하게 하나의 전환 응답으로 대표하기 어렵다. [전환 파형](control_characteristics/transitions.png) · [전환별 집계](control_characteristics/transitions.csv)

## E08–E11, E13–E14, E16–E17 민감도

| 실험 | 측정 결과와 적용 범위 |
|---|---|
| **E08 접촉 solver·형상** | `wz=+1`의 50–60초 잔류 `vx`는 `impratio 100/10/1`에서 `0.10075/0.0796/0.00542 m/s`다. 동일 첫 관측·목표·토크에서 `100→1`은 첫 틱 법선 하중을 최대 `0.560 N` 바꿨다. 실제 관절 상태를 쓰는 토크는 `5 ms`, actor·target은 `20 ms`에 차이가 났다. 초기 roll `+1°/−1°/+0.5°` 3쌍에서 `vx` MAE 변화는 평균 `−0.09533 m/s`, 관측 표준편차 `4.63e−8 m/s`였다. 이는 3개 결정론적 짝이며 DR 모집단 분산의 추정치는 아니다. Solver 최대 반복 `100→20` 효과는 작았다. Line search `50→10`은 `impratio=100`에서 signed `vx`를 `0.0170`으로 낮췄으나 `vx/vy/wz` MAE는 `0.0650/0.1139/0.1524`였다. `impratio=1`에서는 같은 line search 변경의 응답 차가 작았다. Cylinder 형상은 별도 실험에서 `vx=0.1065`였다. [최초 분기](control_characteristics/causal_onset.csv) · [좌회전 신호](control_characteristics/turn_causal_chain.png) · [짝 통계](control_characteristics/seed_effects.csv) |
| **E09 gain** | 좌회전 다리 Kp `0.8/1.2×`는 10–20초 `vx` MAE를 기준 `0.0976`에서 `0.0325/0.0714 m/s`로 바꿨고 yaw MAE는 각각 `0.0467/0.0481 rad/s`로 증가했다. Kd 단독 `±20%`의 정상상태 효과는 작았다. 고속 전진 Kd 단독은 앞의 낮은 target 결론을 바꾸지 않았다. [지표](control_characteristics/metrics.csv) |
| **E10 세 주기** | 계측 스케줄러의 `(50,200,200) Hz`는 기존 경로와 관측·action·상태·target·토크가 일치했다. 좌회전에서 물리만 `200/400/1000 Hz`, PD만 `100/200/500 Hz`로 바꾼 조건의 10–20초 `vx` MAE는 약 `0.098`이었다. 동일 PD·물리에서 actor를 `25 Hz`로 내리면 yaw MAE가 `1.096 rad/s`로 커졌다. 정해진 정책 갱신 간격에 대한 민감성이며, 주파수 변경의 효과를 이 좌회전 이외로 일반화하기 어렵다. [주기별 조건과 지표](control_characteristics/run_index.csv) · [시간 정렬 검증](control_characteristics/analysis_final.log) |
| **E11 합성 지연** | target·actor 센서·PD 센서·토크에 각각 5/10/20 ms를 주입했다. 이 좌회전에서 앞 세 경로의 20 ms `vx` MAE는 `0.0973–0.0977 m/s`였다. 최종 torque의 20 ms는 `vx` MAE를 `0.0245`로 바꿨으나 yaw MAE가 `0.0248→0.0464 rad/s`로 늘었다. 평균 10 ms torque jitter(5/10/15 ms)의 마지막 창 값은 고정 10 ms와 가까웠다. 이 결과는 물리 틱 큐의 개입이며 ROS executor·CAN·센서 통신의 시간 분포는 측정하지 않았다. [지연별 지표](control_characteristics/metrics.csv) |
| **E13 물리량** | `+1.2 m/s`에서 유효 마찰 `μ=0.2`는 공칭과 초기 roll 3개 모두 base 접촉으로 `0.635–0.660 s`에 조기 종료됐다. `μ=0.4,0.6,0.8,1.6`은 완료했다. 좌회전 `μ=0.2`에서는 `impratio=100/1`의 signed `vx`가 각각 `−0.0410/−0.0406`으로 가까웠다. 따라서 공칭 마찰에서의 solver 효과는 마찰에 의존한다. 질량·COM 3축·관성은 좌회전에서 한 파라미터씩 변경했다. 저장 DR 전체 분포와 결합 고장 확률은 추가 평가 범위다. [실행 목록](control_characteristics/run_index.csv) · [저마찰 종료 기록](control_characteristics/forward_12--mu=0.2/events.csv) |
| **E14 외력** | world y 방향 10 N을 5초부터 0.10초 또는 2초 동안 적용했다. 2초 외력 뒤 무외력 대조 대역으로 1초 복귀하는 데 정지 `1.80 s`, 전진 `0.26 s`가 걸렸다. 정지 피치 최대 편차는 각 시험에서 `3.78°/3.35°`였다. 회복 대역은 분석용이며 초기 방향과 외력 크기에 한정한다. [회복 지표](control_characteristics/recovery.csv) |
| **E16 주파수 응답** | 20초 운전점 정착 후 `0.25–4 Hz`를 가진했다. `vx=0.8±0.1`의 기본파 이득은 `0.25/0.5/1 Hz`에서 `0.689/0.450/0.200`, 위상은 `−46.9°/−78.8°/−124.2°`였다. 같은 조건의 바퀴 target→속도 이득은 0.25 Hz에서 `0.947`이었다. 최저 측정 이득 대비 −3 dB 교차는 `0.25–0.5 Hz` 구간에 있다. `wz=1±0.15`는 4 Hz에서 이득 `0.891`, 위상 `−35.3°`로 측정 범위 안에서 −3 dB 교차가 없었다. 위상으로 계산한 등가 지연은 주파수별 결과이며 통신 지연과는 별개다. [Bode 그림](control_characteristics/frequency_response.png) · [주파수 표](control_characteristics/frequency_response.csv) |
| **E17 센서** | 저장 관측 800개×18 조건에서 학습 잡음 반폭 1배를 썼을 때 wheel target 변화 RMS는 dq `5.48`, 투영 중력 `3.88`, gyro `1.19 rad/s`였다. 폐루프에서 yaw gyro bias `+0.05 rad/s`는 전진의 signed yaw rate를 약 `+0.0128→−0.0210 rad/s`로 바꿨다. 같은 bias의 공칭 정지에서 마지막 10초 signed `vx`는 `−0.0866 m/s`였다. gyro Gaussian RMS 대조와 5틱 hold dropout도 측정했다. 다른 센서군의 폐루프 주입은 고정 관측의 민감도 결과와 구별한다. [고정 관측 비교](control_characteristics/sensor_probe.csv) · [폐루프 지표](control_characteristics/metrics.csv) |

## E12 다목적 추종과 E18 별도 Teacher 비교

**E12.** 정지에서 height 목표 `0.370 m`의 마지막 10초 높이 MAE는 `4.61 mm`, 같은 목표를 곡선 `(+0.8 m/s, +0.8 rad/s)`에 더하면 `70.89 mm`였다. 다섯 목표를 함께 준 조건은 `vx/wz` MAE `0.0240 m/s / 0.0324 rad/s`, roll/pitch MAE `1.49°/1.49°`, height MAE `71.27 mm`였다. 정지 자세 묶음은 높이 MAE가 `20.44 mm`였지만 pitch MAE가 `6.43°`, 잔류 `vx` MAE가 `0.0783 m/s`였다. 높이 단독과 곡선에 높이만 더한 대조로 이 운전점의 속도–높이 결합을 확인했다. 투영 중력·높이·속도 보상의 일부를 같은 로그에서 재계산했다. 저장 정책의 학습 원인을 이 값만으로 식별할 수는 없다. [다축 응답](control_characteristics/multi_objective.png) · [보상 항 대조](control_characteristics/tracking_reward_terms.csv)

**E18.** Teacher80k+LPF60에 같은 초기 상태·동일 500 Hz 물리를 사용했다. 표에서 상태 오차는 마지막 10초, 변위는 지정 구간의 시작–끝 거리다. 이동 명령 중 residual 요청·필터 토크는 0이었다. 활성 시 요청 peak는 `6.81 N·m` 이내였고 모든 관절을 합친 토크 RMS는 네 페어 모두 줄었다. [ON/OFF 전체 표](control_characteristics/paired_effects.csv) · [Gate·토크 검사](control_characteristics/residual_gate.csv) · [응답 그림](control_characteristics/teacher_residual.png)

| Teacher80k 시험 | OFF | ON |
|---|---:|---:|
| 공칭 정지 30초 변위 | 0.0073 m | 0.0493 m |
| 공칭 정지 높이 MAE | 11.76 mm | 24.39 mm |
| 정지 height 0.370 m의 높이 MAE | 15.50 mm | 1.48 mm |
| 정지 roll +10°에서 roll MAE | 약 3.17° | 약 3.58° |
| 정지→주행→정지의 마지막 20초 변위 | 0.159 m | 0.042 m |

## 재현과 해석 범위

실험 전용 계측은 [실행기](../../../test/evaluate_control_characteristics.py), 창 집계·그림·검사는 [분석기](../../../test/analyze_control_characteristics.py)에 있다. 생산 제어기에는 계측 변경을 넣지 않았다. 개별 폴더의 manifest에는 환경·해시·실행 인자·유효 파라미터가 있고 policy/physics NPZ는 명령, 관측, torques, 물리 GT, 센서 신호와 접촉 시각을 해당 틱으로 연결한다. 소스별 구분과 단위는 결과의 필드 이름과 manifest에 있다. 일부 초기 수집의 적용 시각은 물리 pre-tick 시각으로 재현할 수 있으며 별도의 monotonic 토크 인가 필드는 후속 수집에 추가됐다.

```bash
source /opt/ros/jazzy/setup.bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST OPENBLAS_NUM_THREADS=1
run_root="docs/exp/$(date +%F)/control_characteristics_replay"
~/venvs/tita_wbc_jazzy/bin/python test/evaluate_control_characteristics.py \
  "$run_root/forward_12" forward_12 --parity
~/venvs/tita_wbc_jazzy/bin/python test/evaluate_control_characteristics.py \
  "$run_root/turn_left" turn_left --parity
~/venvs/tita_wbc_jazzy/bin/python test/evaluate_control_characteristics.py \
  "$run_root/sequence" sequence
~/venvs/tita_mjlab/bin/python test/evaluate_control_characteristics.py "$run_root/native" native
```

전체 결과의 재집계는 `~/venvs/tita_mjlab/bin/python test/analyze_control_characteristics.py docs/exp/2026-11-24/control_characteristics`로 실행한다. 해당 명령은 원시 NPZ를 사용하고 시뮬레이션을 진행하지 않는다. 실패한 초기 계측 시도는 `diagnostic_attempts/`에 보관했다. 초기 ROS context 재사용, Native actuator 순서와 관측 캐시, PD 지연 큐의 틱 인덱스를 바로잡고 영향받은 시험을 재실행했다. 이 시도들의 값은 집계에서 제외했다. [분석 검증 로그](control_characteristics/analysis_final.log) · [저장소 검사](control_characteristics/agent-check.log)

관측된 제한은 해당 정책과 운전 조건에 적용된다. 정지·직진 경로를 평가하려면 속도 오차에 절대 위치·heading 오차를 추가해야 한다. 넓은 DR 조합, 외력 방향과 크기 전수, 센서군 전체의 폐루프 교차변화, 실제 ROS/CAN 지연은 미검증 범위다. 실기에서는 동기화한 IMU·encoder·CAN 시각, 모터 전류 및 구동기 리포트, 외부 위치·heading 기준계가 필요하다. 이 자료가 있어야 시뮬레이션의 정책·접촉 원인을 실제 플랫폼의 상태 추정과 액추에이터 응답에 대조할 수 있다.
