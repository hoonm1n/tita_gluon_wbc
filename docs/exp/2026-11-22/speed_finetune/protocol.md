# 속도 추종 미세학습 실험 조건

실행 도메인 TRAINING_SIM, VALIDATION_SIM. 원본은 flat_roll_pitch_height/model_85000. Actor SHA256 `011c043642752e3e4adc51f8d401d9dbbce8a9d2d98948f813358fde6eb893a5`. 기준선 재현 PASS.

- A는 기존 설정, B는 명령 분포 변경, C는 가장 좁은 평면 속도 보상의 가중치 0.5→1.0, D는 두 변경을 적용한다. B·D의 재추출 비율은 정지 10%, 고속 직선 15%, 순수 회전 15%, 혼합 주행 60%다. 고속 직선의 절댓값은 1.0–1.5 m/s, 순수 회전은 0.4–1.6 rad/s다. 각 부호의 확률은 절반이다.
- 모든 학습은 같은 원본에서 warm start로 초기화한다. 환경 4096개, seed 42·43·44, 10,000 벡터 스텝, 리플레이 용량 100만, 최소 10만, 배치 2048, 스텝당 업데이트 2회다. 5,000스텝에서는 sim2sim 두 접촉 설정의 ±1.2 m/s 및 제자리 ±1 rad/s를 진단한다. 채택에는 10,000스텝을 사용한다. 기본 학습기와 실험 스크립트가 로컬에 메트릭을 기록한다.
- 기준선 재현에서 sim2sim 기존 58종은 impratio 100에서 49개, 1에서 55개가 통과했다. 각 설정에서 추가 자세 시험 4종을 실행했다. native mjlab은 고속 직선 4종, 순수 회전 6종, 곡선 2종, 자세 시험 4종을 수행했다. 모든 평가 시간은 60초다. [기준선 구간별 지표](source_windows.csv)
- 후보별로 원본과 같은 조건을 실행한다. 몸체 기준 축별 MAE와 부호 있는 오차, 실제·목표 바퀴 속도, 축 중앙의 전진 속도, 자세·높이 오차를 집계한다. 제어 주기에서 관측된 차체·무릎 접촉을 전 구간 집계한다. native 평가에서는 접촉 후에도 점수를 기록하기 위해 접촉 종료만 끈다.
- 공칭 상태의 마지막 10초 동안 세 평가 환경 모두에서 +1.2/+1.4 m/s의 vx MAE ≤0.05/0.08 m/s, 모든 제자리 회전의 vx MAE ≤0.05 m/s와 yaw rate MAE <0.10 rad/s를 요구한다. 기존 58개 중 각 설정에서 원본이 통과한 조건을 모두 보존해야 한다. 자세 시험의 명령축/높이 MAE 증가는 0.5–60초와 50–60초 모두에서 1°/0.01 m 이하여야 한다. 60초 완료와 접촉 0회를 요구한다. 같은 군에서 3개 seed 중 2개 이상이 이 조건을 모두 만족해야 채택한다.
- 채택 가능한 군이 여럿이면 seed마다 sim2sim 두 설정에서 고속 전진 및 순수 회전의 허용값 대비 최대 오차를 구하고 군별 중앙값이 가장 작은 군을 선택한다. 해당 군에서 실제로 통과한 seed 중 이 비율이 가장 작은 체크포인트를 제시한다.

한 학습군·seed를 재현하는 예시다.

```bash
arm=B
seed=42
~/venvs/tita_mjlab/bin/python -m mj_lab.scripts.train --task flat_roll_pitch_height --headless \
  --warm_start mj_lab/logs/flash_sac/heading_baseline/model_85000 --speed-ablation "$arm" \
  --run_name "speed_${arm}_${seed}" --num_envs 4096 --steps 10000 --seed "$seed" \
  --save_every 5000 --log_every 100 --video_interval 0 --logger console
```

평가 실행기는 각 학습의 저장 완료를 확인한 후 ONNX 내보내기, sim2sim, native 순서로 평가한다. 원본을 동일한 방법으로 평가한 source 폴더를 사용한다.

```bash
out=docs/exp/2026-11-22/speed_finetune
source /opt/ros/jazzy/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_LOG_DIR="$PWD/$out/ros_logs"
unset ROS_LOCALHOST_ONLY
~/venvs/tita_mjlab/bin/python test/run_speed_finetune_eval.py "$out"
~/venvs/tita_mjlab/bin/python test/report_speed_finetune.py "$out"
```
