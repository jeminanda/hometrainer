# preprocessing

키포인트 시퀀스를 학습 가능한 형태로 가공하는 모듈.

## 파이프라인 (build_dataset.py가 이 순서로 전부 이어붙임)

1. **마스킹/보간** — visibility가 낮은 관절의 (x,y)를 NaN 처리 후 시간축 선형 보간.
   영상 전체에서 한 번도 신뢰할 수 없었던 관절(예: 팔굽혀펴기에서 화면 밖 무릎)은
   보간 기준점이 없으므로, 원본의 저신뢰 좌표값으로라도 채워 넣는 폴백이 있습니다.
2. **좌표 정규화** (`coordiante_normalization.py`, 파일명 오타 그대로 유지) — 골반 중앙점
   기준 상대 좌표 + 몸통 길이(어깨중심-힙중심) 기준 스케일 정규화. 프레임별로 따로
   정규화하지 않고 시퀀스 전체의 중앙값 스케일 하나로 고정해, 영상 재생 중 몸 크기가
   출렁이지 않게 합니다.
3. **신뢰도 높은 쪽 선택** (`angles.py`의 `select_reliable_side`) — 측면 촬영에서는
   카메라에서 먼 쪽 팔/다리가 구조적으로 가려지는 경우가 흔합니다(실측 확인됨). 영상
   전체 기준으로 visibility가 더 높은 쪽만 골라 그 쪽 각도로 계산합니다. **좌우 비대칭
   검사는 이 구조상 지원하지 않습니다.**
4. **rep 슬라이싱** (`rep_slicer.py`) — 관절 각도 시계열의 극소점(valley, 최대 굴곡 지점)을
   찾아 그 사이 구간을 rep 하나로 자릅니다. 극소점이 1개 이하면 영상 전체를 rep 1개로
   간주하는 폴백이 있습니다(단, 실제로 유의미한 가동범위가 있을 때만). `min_distance`
   (기본 30)가 너무 낮으면 동작 중 잠깐 멈칫하는 습관이 있는 영상에서 하나의 rep이
   여러 조각으로 잘못 쪼개지는(과다 분절) 문제가 생길 수 있습니다.
5. **DTW 위상 정규화** (`phase_normalization.py`) — 가변 길이 rep을 고정 길이(기본 100
   프레임)로 맞출 때, 단순히 프레임을 균등하게 늘리지 않고 **모범 사례 평균 궤적(템플릿)에
   Dynamic Time Warping으로 정렬**합니다. 사람마다 다른 템포(예: 바닥에서 오래 머무는
   스타일)가 있어도, 그 템포 차이를 정상적인 변동으로 흡수하기 위함입니다 (실측으로
   템포 차이 때문에 정상 폼이 이상치로 잘못 잡히던 문제를 해결함).
   - `build_template()`: 여러 rep의 각도 시계열 평균으로 템플릿 생성
   - `normalize_phase_dtw()`: 개별 rep을 템플릿에 맞춰 정렬
   - 예전 방식(`normalize_phase()`, 단순 선형 보간)도 템플릿 생성용 부트스트랩과
     하위 호환을 위해 남아있습니다.
6. **특징 추출** (`feature_extraction.py`) — 관절별로 `min_angle_*`/`max_angle_*`/`rom_*`
   (요약 통계) + `{joint}_bin00~09`(구간별 평균 각도, 어느 phase에서 문제가 생겼는지
   식별용) + `reliable_side_*`(메타데이터, 학습용 숫자 특징에서는 제외됨)를 반환합니다.

## 시각화

- `visualizer.py`의 `animate_skeleton_2d()` — 정규화된 좌표만으로 그린 2D 애니메이션(gif/mp4)
- `overlay_visualizer.py`의 `overlay_skeleton_on_video()` — **원본 영상 위에 실제 픽셀
  좌표로 스켈레톤을 그려서** 오버레이 영상을 만듭니다. 반드시 정규화 **전** 원본
  keypoints(`BlazePoseExtractor` 출력 그대로)를 넣어야 합니다.

## build_dataset 실행

```bash
python -m src.preprocessing.build_dataset \
    --raw-dir data/raw --output-dir data/processed \
    --target-length 100 --min-distance 30 --min-rep-frames 30
```

`data/raw/{exercise}_{video_id}_keypoints.npy` 명명 규칙을 따르는 파일을 모두 순회합니다
(규칙에 안 맞으면 `--manifest`로 csv 지정 가능).

### 2-패스 구조 (DTW 도입으로 인한 변경)

DTW는 템플릿(모범 사례 평균 궤적)이 먼저 있어야 하고, 그 템플릿은 전체 rep을 한 번 다
봐야 만들 수 있습니다. 그래서:
1. **1패스**: 모든 파일에서 rep 구간(가변 길이)까지만 추출해 전부 모음 → exercise별 템플릿 생성
2. **2패스**: 각 rep을 해당 exercise 템플릿에 DTW로 맞춰 위상 정규화 + 특징 추출

새 영상 1개만 채점하려는 경우(전체 재학습 없이), `process_source_file_with_template()`로
저장된 템플릿(`templates.npz`)을 재사용하면 됩니다 — 학습/채점에 서로 다른 템플릿을
쓰면 기준이 어긋나므로 반드시 같은 템플릿을 재사용해야 합니다.

## 출력 (data/processed/)

- `rep_sequences.npy` — `(전체 rep 수, target_length, 33, C)` DTW로 위상 정규화된 rep 시퀀스
- `rep_features.csv` — rep별 메타데이터(video_id, exercise, rep_idx, seq_index) + 특징
- `templates.npz` — exercise별 DTW 템플릿 (채점 시 재사용)
- `build_log.json` — 파일별 성공/실패 로그

## EXERCISE_JOINTS (angles.py)

운동별로 추적하는 관절 삼각형(각도 계산용)을 정의합니다:
- `squat`: `knee` (hip-knee-ankle)
- `pushup`: `elbow`(shoulder-elbow-wrist, rep 슬라이싱 기준 관절이라 항상 첫 번째로 유지),
  `hip`(shoulder-hip-ankle, 몸통 일직선/처짐), `knee`(hip-knee-ankle, 다리 일직선),
  `shoulder`(elbow-shoulder-hip, 팔꿈치 벌어짐)

새 운동을 추가하려면 여기에 관절 정의만 추가하면, `feature_extraction`이 자동으로
그 관절들의 특징을 뽑습니다 (다른 코드 수정 불필요).
