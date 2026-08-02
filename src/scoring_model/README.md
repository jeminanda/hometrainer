# scoring_model

`preprocessing`이 만든 특징으로 "이 rep이 정상적인 폼인지"를 0~100점으로 채점하는 모듈.

## 왜 Mahalanobis distance인가

가진 데이터가 전부 "모범 사례"(정상 수행)뿐이고 오답 라벨이 없습니다. 이런 경우
지도학습 분류기(SVM 등) 대신, "정상 사례들의 분포에서 얼마나 벗어났는가"를 재는
**Mahalanobis distance 기반 이상탐지**가 적합합니다. 표본이 소수(4~13명 수준)의
영상에 몰려있는 경우가 많아, 공분산 추정은 **Ledoit-Wolf shrinkage**를 기본으로 씁니다.

## 파일

| 파일 | 역할 |
|---|---|
| `reference_stats.py` | `fit_reference()`로 정상 사례 특징 행렬 → 평균/공분산 학습, `ReferenceStats` 저장/로드 |
| `scorer.py` | `mahalanobis_distance()`, `distance_to_score()`(0~100점 변환), `per_feature_contribution()`(어느 특징이 문제인지 z-score), `score_rep()` |
| `rule_based.py` | 통계만으로는 못 잡는 "방향성 있는" 특징(깊이 등)에 대한 규칙 기반 최소점수 보정 |
| `train.py` | `rep_features.csv` → exercise별 `{exercise}_reference.npz` 학습/저장 (CLI 겸용) |
| `score_reps.py` | 새 영상 채점(`score_video`), decay_scale 역산(`calibrate_and_rescore_video`), CLI |
| `normalization.py` | (현재 미사용) 영상별 중앙값 대비 편차 특징 — DTW로 템포 문제가 해결되면서 롤백됨. 필요해지면 `train_for_exercise`에 다시 연결 |

## 점수 산출 방식 (scorer.py)

`distance_to_score(distance, stats, method="threshold")`:
- 기준 집단의 상위 `inside_percentile`(기본 95%) 이내 거리면 **100점**
- 벗어나면 `decay_scale`(기본 3.0)로 완만하게 감쇠: `100 * exp(-초과분/decay_scale)`
- 경계를 `max()`가 아니라 백분위로 잡는 이유: 기준 데이터 자체에 노이즈(과다 분절 등)가
  섞여도 그 이상치 하나가 "만점 구간"을 왜곡시키지 않도록 하기 위함

`calibrate_decay_scale(distance, stats, target_score)` — "이 rep은 이 정도 점수였으면
좋겠다"를 역산해서 `decay_scale`을 구합니다. **주의: 이 값을 전역 기본값으로 승격하지
마세요** — 특정 영상 하나를 보고 역산한 값이라, 다른 모든 영상/운동에 똑같이 적용하면
진짜 이상치까지 관대해지는 부작용이 있습니다. `score_video`/`calibrate_and_rescore_video`로
그 영상 하나만 조정하는 용도로 쓰세요.

## 규칙 기반 보정 (rule_based.py)

Mahalanobis는 "기준 집단 평균에 가까울수록 좋다"는 전제인데, 스쿼트/팔굽혀펴기 깊이처럼
"평균이 아니라 특정 방향(더 깊이)일수록 좋은" 특징에는 이 전제가 안 맞습니다. 실제로
기준 영상 평균보다 훨씬 깊게 앉는 사람이 오히려 낮은 점수를 받는 사례가 확인되어 도입했습니다.

- **게이트형 규칙** (`squat`의 knee, `pushup`의 elbow): 일정 깊이 이상 굽혀야만 작동.
  통과하면 기본 50점에서, 펴짐(top) 부족 정도에 따라 최대 25점까지 **가속형(exponential)**
  으로 감점 (부족분이 작으면 거의 안 깎이다가 커질수록 급격히 깎임).
- **pushup의 hip/knee(몸통·다리 일직선)**: 게이트 없이, elbow가 만든 기본점수에서
  각각 최대 12.5점씩 추가로 깎는 감점 항목으로 통합.
  (처음엔 elbow/hip/knee를 각각 따로 계산해 `max()`로 묶었는데, elbow만 좋아도
  hip/knee가 나빠도 점수가 구제되는 버그가 있어서 지금 방식으로 바꿈.)
- **최종 점수 = `max(mahalanobis_score, 규칙_score)`** — mahalanobis가 이미 잘 준 점수는
  그대로 유지하고, 규칙은 "구제용 바닥"으로만 작동합니다.
- `score_source`("rule" 또는 "mahalanobis") 필드로 최종 점수가 어느 쪽에서 나왔는지
  확인할 수 있습니다. `"rule"`인 경우, `top_issues`(mahalanobis 기준 z-score)가 실제
  감점/구제 사유와 다를 수 있다는 점을 유의하세요.
- 팔꿈치 벌어짐(`shoulder`)처럼 "적정 범위가 있고 방향성이 불명확한" 특징은 규칙화하지
  않고 통계(mahalanobis)로만 남겨뒀습니다.

## 학습

```bash
python -m src.scoring_model.train \
    --processed-dir data/processed --exercise pushup --output-dir models
```

## 새 영상 채점

```python
from src.scoring_model.score_reps import score_video, load_template

template = load_template("data/processed", "pushup")
results = score_video(
    keypoints_path="tests/new_video_keypoints.npy",
    exercise="pushup",
    template=template,
    model_dir="models",
)
```

또는 `notebooks/score_new_video.ipynb`로 영상 입력부터 점수·자연어 피드백까지 한 번에
확인할 수 있습니다 (원본 영상만 있으면 좌표 추출부터, 이미 추출된 `.npy`가 있으면
전처리부터 자동으로 진행).

## 검증

`notebooks/validate_scoring_model.ipynb`가 hold-out(학습에 안 쓴 정상 rep) + 의도적으로
폼이 나쁜 예시로 good/bad 그룹을 나눠 점수 분포를 비교하고(Mann-Whitney U 검정),
`top_issues`가 의도한 오류 유형과 실제로 일치하는지 확인합니다. **`data/raw_test`
같은 검증용 데이터로 별도 `build_dataset()`을 다시 호출하면 안 되고**, 반드시
`data/raw`(기준)로 만든 템플릿을 재사용해야 학습/검증 기준이 일치합니다.
