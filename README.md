# Home Training Scorer

비전 AI를 활용해 홈 트레이닝 동작(팔굽혀펴기, 스쿼트 등)의 수행도를 자동으로 채점하는 프로젝트입니다.

## 프로젝트 개요

1. 영상에서 BlazePose(mediapipe Tasks API)로 관절 키포인트 좌표를 추출
2. 반복 동작(rep) 단위로 슬라이싱하고, DTW(Dynamic Time Warping)로 속도(템포) 차이를 정규화
3. 관절 각도 기반 특징을 뽑아, "정상 사례만 있는" 데이터에 맞는 Mahalanobis distance 기반 채점 + 방향성 있는 특징(깊이 등)에 대한 규칙 기반 보정을 결합해 점수 산출

> 촬영 각도는 **측면(side view)** 을 전제로 합니다 — z좌표 없이 2D 각도만 쓰기 때문에, 정면에 가까운 각도로 찍으면 관절 각도가 실제보다 얕게(펴진 것처럼) 측정되는 투영 왜곡이 생깁니다. 데이터 수집 시 촬영 각도를 확인해주세요.
>
> 엣지 디바이스(모바일) 포팅은 후순위 목표로, 현재는 포즈 추출 → 전처리 → 채점 모델 개발에 집중합니다.

## 폴더 구조

```
home-training-scorer/
├── data/
│   ├── raw/            # 원본 영상 / 원본 키포인트 추출 결과 (학습용 "모범 사례"만 모아둠)
│   └── processed/      # rep_sequences.npy, rep_features.csv, templates.npz (build_dataset 출력)
├── models/              # exercise별 학습된 기준 통계 (squat_reference.npz, pushup_reference.npz 등)
├── src/
│   ├── pose_extraction/ # BlazePose(mediapipe)로 키포인트 추출
│   ├── preprocessing/   # 마스킹/보간, 정규화, rep 슬라이싱, DTW 위상정규화, 특징 추출
│   └── scoring_model/   # Mahalanobis 기반 통계 채점 + 규칙 기반 보정, 학습/추론 코드
├── notebooks/           # 탐색적 분석, 검증, 시연용 노트북 (Test.ipynb, score_new_video.ipynb 등)
├── tests/               # 시연/검증용 영상 및 산출물 (data/raw와 분리 — 학습에는 안 씀)
├── requirements.txt
└── README.md
```

각 하위 모듈의 상세 설명은 `src/pose_extraction/README.md`, `src/preprocessing/README.md`,
`src/scoring_model/README.md`를 참고하세요.

## 파이프라인

```
원본 영상 (mp4, 측면 촬영)
   │
   ▼
[pose_extraction] BlazePoseExtractor (mediapipe Tasks API, GPU delegate 지원)
   │  → 원본 키포인트 시퀀스 (T, 33, C) — C는 (x,y,visibility) 또는 (x,y,z,visibility)
   ▼
[preprocessing] build_dataset()
   │  1) 마스킹/보간 (저신뢰 관절 결측 처리)
   │  2) coordiante_normalization: 힙 중심 정렬 + 몸통 스케일 정규화
   │  3) select_reliable_side: 측면 촬영으로 가려지는 관절은 좌/우 중 신뢰도 높은 쪽만 사용
   │  4) rep_slicer: 관절 각도 극소점 기준 rep 슬라이싱
   │  5) DTW 위상 정규화: 기준 영상 평균 궤적(템플릿)에 맞춰 정렬 (템포 차이에 안 흔들림)
   │  6) feature_extraction: min/max/rom + 구간별(phase-bin) 각도 특징 추출
   │  → rep_sequences.npy, rep_features.csv, templates.npz
   ▼
[scoring_model]
   │  - reference_stats.fit_reference: "모범 사례"만으로 평균/공분산(Ledoit-Wolf shrinkage) 학습
   │  - scorer: Mahalanobis distance → threshold 기반 점수 (기준 상위 95% 이내는 100점)
   │  - rule_based: 깊이처럼 "평균이 아니라 특정 방향일수록 좋은" 특징에 대한 최소점수 보정
   │    (mahalanobis와 규칙 점수를 max()로 결합)
   ▼
동작 점수(0~100) + 어느 관절/구간이 문제인지(top_issues) + 자연어 피드백
```

## 알려진 제약 사항

- **측면 촬영 전제**: z좌표 없이 2D 각도만 쓰므로, 정면에 가까운 각도는 각도 왜곡을 일으킵니다.
- **표본이 적은 영상(rep 1~3개)은 통계적으로 불안정**합니다. 가능하면 반복을 충분히(5회 이상) 포함해 촬영해주세요.
- **rep 슬라이싱은 valley-to-valley(저점-저점) 방식**이라, 영상 맨 앞/맨 뒤의 반쪽 동작은 자동으로 버려집니다 (저점이 N개면 rep은 N-1개).
- 좌우 비대칭 검사는 현재 지원하지 않습니다 (`select_reliable_side`가 신뢰도 높은 한쪽만 쓰는 구조라서).

## 로드맵

- [x] 포즈 키포인트 추출 파이프라인 구축 (BlazePose, GPU delegate)
- [x] rep 슬라이싱 및 위상 기반(DTW) 정규화 구현
- [x] 베이스라인 채점 모델(Mahalanobis + 규칙 기반) 개발
- [x] 새 영상 채점 + 원본 영상 위 스켈레톤 오버레이 시각화
- [ ] (확장) 키포인트 시퀀스 기반 Transformer 채점 모델
- [ ] (확장) 시점(카메라 각도) 불변 특징 — `pose_world_landmarks` 검증, 뷰-불변 임베딩 등
- [ ] (후순위) 엣지 디바이스(모바일) 경량화 및 포팅

## 개발 환경

```
pip install -r requirements.txt
```

BlazePose 모델(.task) 다운로드:
```
wget -O models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```

## 라이선스

TBD
