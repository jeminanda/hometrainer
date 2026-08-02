"""
src/scoring_model/rule_based.py

Mahalanobis(통계) 채점이 놓치는 "방향성 있는" 특징(스쿼트/팔굽혀펴기 깊이, 몸통/다리
일직선 여부)에 대해 최소 점수를 보장하는 규칙 기반 채점.

배경: Mahalanobis distance는 "기준 집단 평균에 가까울수록 좋다"는 전제인데, 깊이처럼
"평균이 아니라 특정 방향(더 깊이)일수록 좋다"는 특징에는 이 전제가 안 맞는다.
실제로 기준 영상 평균보다 훨씬 깊게 앉는 사람이 오히려 낮은 점수를 받는 사례가 있었다.

설계:
- 관절별로 규칙 점수(0~50점, 조건 불충족 시 0점)를 계산하고
- 최종 점수는 `max(모든 규칙 점수, mahalanobis 점수)`로 합친다
  -> mahalanobis가 이미 잘 준 점수는 그대로 유지하면서, "방향성 있는 좋은 동작인데
     기준 집단 평균과 달라서 낮게 나온" 경우만 규칙이 구제한다.

관절별 규칙:
- squat 무릎 / pushup 팔꿈치(굽힘 관절): 일정 깊이 이상 굽혀야(게이트) 규칙이 작동한다.
  통과하면 50점 기본 + 펴짐 부족 정도에 따라 최대 25점까지 가속(exponential) 감점
  (결과 25~50점 사이). 깊이 게이트를 못 넘으면(충분히 안 굽혔으면) 0점 — 규칙이
  관여하지 않고 mahalanobis 점수가 그대로 쓰인다.
- pushup의 hip/knee(몸통/다리 일직선): 게이트 없이, 180도에서 벗어난 정도로
  최대 25점까지 가속 감점 (결과 25~50점 사이).
"""

from __future__ import annotations

import numpy as np

from .scorer import score_rep


def accelerating_penalty(deficit: float, width: float, max_penalty: float = 25.0, k: float = 4.0) -> float:
    """
    deficit(목표 대비 부족한 정도)이 0이면 감점 0, width에 도달하면 정확히 max_penalty.
    k가 클수록 부족분이 width에 가까워질수록 감점이 더 급격해진다
    ("너무 안 펴져 있으면 점수가 확 깎이는" 가속형 곡선). k=4가 기본값.
    """
    if deficit <= 0:
        return 0.0
    if deficit >= width:
        return max_penalty

    raw = np.exp(k * deficit / width) - 1.0
    raw_max = np.exp(k) - 1.0
    return float(max_penalty * raw / raw_max)


def gated_rule_score(
    depth_value: float,
    depth_gate: float,
    extension_value: float,
    extension_target: float,
    penalty_width: float,
    base_score: float = 50.0,
    max_penalty: float = 25.0,
    k: float = 4.0,
) -> float:
    """
    "일정 깊이(depth_gate) 이상 굽혀야만 작동"하는 게이트형 규칙.
    depth_value가 depth_gate보다 크면(=충분히 안 굽혔으면) 0점.
    통과하면 base_score(50점)에서 시작해, extension_target 대비 extension_value가
    부족한 만큼 accelerating_penalty로 최대 max_penalty(25점)까지 감점한다.
    """
    if depth_value > depth_gate:
        return 0.0
    deficit = max(0.0, extension_target - extension_value)
    penalty = accelerating_penalty(deficit, penalty_width, max_penalty=max_penalty, k=k)
    return base_score - penalty


def ungated_rule_score(
    value: float,
    target: float,
    penalty_width: float,
    base_score: float = 50.0,
    max_penalty: float = 25.0,
    k: float = 4.0,
) -> float:
    """
    게이트 없이, target(예: 180도, 일직선)에서 value가 벗어난 만큼 accelerating_penalty로
    최대 max_penalty(25점)까지 감점하는 규칙 ("깊이" 개념이 없는 몸통/다리 일직선 등에 사용).
    """
    deficit = max(0.0, target - value)
    penalty = accelerating_penalty(deficit, penalty_width, max_penalty=max_penalty, k=k)
    return base_score - penalty


def squat_rule_score(
    features: dict, depth_gate: float = 105.0, extension_target: float = 170.0, penalty_width: float = 40.0
) -> float:
    """squat: 무릎 깊이 게이트(105도) + 펴짐(170도 목표) 가속 감점(폭 40도)."""
    return gated_rule_score(
        depth_value=features["min_angle_knee"],
        depth_gate=depth_gate,
        extension_value=features["max_angle_knee"],
        extension_target=extension_target,
        penalty_width=penalty_width,
    )


def pushup_rule_score(
    features: dict,
    elbow_depth_gate: float = 90.0,
    elbow_extension_target: float = 170.0,
    elbow_penalty_width: float = 40.0,
    body_line_target: float = 180.0,
    body_line_penalty_width: float = 20.0,
) -> float:
    """
    pushup 규칙 점수. elbow(팔꿈치 깊이 게이트 90도 + 펴짐 170도 목표, 최대 25점 감점)가
    기본 점수를 만들고, hip/knee(몸통/다리 일직선, 180도 목표, 게이트 없음)는 거기서
    각각 최대 12.5점씩 추가로 깎는 감점 항목으로 통합한다.

    (처음엔 elbow/hip/knee를 각각 따로 계산해 max()로 묶었는데, 그러면 elbow만
    좋아도 hip/knee가 아무리 나빠도 elbow 점수가 그대로 이겨서 hip/knee 규칙이
    사실상 무력화되는 버그가 실측으로 확인됐다. 그래서 elbow를 기본점으로 삼고
    hip/knee는 거기서 깎기만 하는 방식으로 바꿨다 — hip/knee가 나쁘면 elbow가
    아무리 좋아도 반드시 점수가 깎인다.)

    elbow 깊이 게이트를 못 넘으면(충분히 안 굽혔으면) 전체 규칙점수는 0점
    (규칙이 관여하지 않고 mahalanobis 점수만 쓰인다).
    """
    min_angle_elbow = features["min_angle_elbow"]
    if min_angle_elbow > elbow_depth_gate:
        return 0.0

    elbow_deficit = max(0.0, elbow_extension_target - features["max_angle_elbow"])
    elbow_score = 50.0 - accelerating_penalty(elbow_deficit, elbow_penalty_width, max_penalty=25.0)

    hip_deficit = max(0.0, body_line_target - features["min_angle_hip"])
    hip_penalty = accelerating_penalty(hip_deficit, body_line_penalty_width, max_penalty=12.5)

    knee_deficit = max(0.0, body_line_target - features["min_angle_knee"])
    knee_penalty = accelerating_penalty(knee_deficit, body_line_penalty_width, max_penalty=12.5)

    return max(0.0, elbow_score - hip_penalty - knee_penalty)


RULE_SCORERS = {
    "squat": squat_rule_score,
    "pushup": pushup_rule_score,
}


def score_rep_with_rules(
    features: dict, stats, exercise: str, decay_scale: float = 3.0, top_k_feedback: int = 2
) -> dict:
    """
    score_rep()(mahalanobis 기반)과 exercise별 규칙 점수를 max()로 합쳐서 채점.

    features: extract_rep_features()가 만든 dict (rep_features.csv의 한 행이어도 됨) —
        stats.feature_names에 대응하는 값들 + min_angle_knee 같은 원본 각도값들이
        모두 들어있어야 한다.

    Returns:
        score_rep()과 동일한 dict에 "score_source"("rule" 또는 "mahalanobis")가 추가된 것.
        규칙이 최종 점수를 결정했다면 "rule"이라, top_issues(mahalanobis 기준 z-score)가
        이 경우엔 실제 감점 사유와 안 맞을 수 있다는 걸 유의할 것.
    """
    feature_vector = np.array([features[name] for name in stats.feature_names])
    result = score_rep(feature_vector, stats, top_k_feedback=top_k_feedback, decay_scale=decay_scale)

    rule_fn = RULE_SCORERS.get(exercise)
    if rule_fn is None:
        result["score_source"] = "mahalanobis"
        return result

    rule_score = rule_fn(features)
    if rule_score > result["score"]:
        result["score"] = rule_score
        result["score_source"] = "rule"
    else:
        result["score_source"] = "mahalanobis"
    return result
