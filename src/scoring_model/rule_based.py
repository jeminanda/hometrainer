"""
src/scoring_model/rule_based.py

Mahalanobis(통계) 채점만으로는 놓칠 수 있는 "최소 동작 기준"에 대해 최저점을
보장하는 규칙 기반 채점.

배경: Mahalanobis distance는 정상 사례 집단과의 통계적 거리로 채점하는데, 표본이
소수의 영상에 몰려있거나 그 사람의 동작이 기준 집단과 여러 면에서 통계적으로
크게 다르면, **최소한의 동작 기준(예: 일정 깊이 이상 굽힘)은 충분히 지켰는데도**
distance가 극단적으로 커져서 거의 0점에 가까운 점수가 나올 수 있다 (실측: 합성
테스트 데이터로 확인한 사례에서 mahalanobis 단독 0.1점).

**주의**: 이 규칙은 "깊게 앉으면 보상해준다"는 게 아니다. 실제로 확인한 깊은
스쿼트 사례(기준 평균보다 훨씬 깊게 앉는 두 영상)는 mahalanobis만으로도 이미
90점대(94.6, 98.3)가 나왔고, 규칙이 개입할 필요도 없었다. 규칙이 실제로 의미를
갖는 상황은 **"최소 동작 기준은 통과했는데 통계 점수가 (이유를 불문하고) 극단적으로
낮게 나온" 경우**뿐이다. 즉 규칙은 "잘하면 더 준다"가 아니라 "최소한은 했으면
바닥까지는 안 떨어지게 막아준다"는 안전망(floor)이다.

설계:
- 관절별로 규칙 점수(0~50점, 조건 불충족 시 0점)를 계산하고
- 최종 점수는 `max(모든 규칙 점수, mahalanobis 점수)`로 합친다
  -> mahalanobis가 이미 더 높은 점수를 줬다면 규칙은 아무 영향도 안 준다.
     mahalanobis가 최소 동작 기준 통과 대비 비정상적으로 낮게 나왔을 때만
     규칙이 "최소 이 정도는 보장"하는 바닥 역할을 한다.

관절별 규칙 (선형 감점):
- squat 무릎 / pushup 팔꿈치(굽힘 관절): 일정 깊이 이상 굽혀야(게이트) 규칙이 작동한다.
  통과하면 50점 기본 + 펴짐 부족 정도에 "선형으로" 비례해 최대 25점까지 감점
  (결과 25~50점 사이). 깊이 게이트를 못 넘으면(충분히 안 굽혔으면) 0점 — 규칙이
  관여하지 않고 mahalanobis 점수가 그대로 쓰인다.
  (처음엔 가속형(exponential) 감점을 썼는데, k=8로 잘못 설정된 채 배포돼서 실측
  결과 각도가 7~8도나 부족해도 감점이 0.03점 수준으로 거의 안 먹히는 문제가
  발견됐다. k를 다시 낮춰도 "조금 부족한 건 거의 안 깎고 많이 부족해야 급격히
  깎는다"는 모양 자체가 튜닝하기 까다롭고 직관적이지 않아서, 선형 감점으로
  단순화했다 — deficit=7.7도, width=40 기준으로 약 4.8점 감점이 나오도록 검증함.)
- pushup의 hip/knee(몸통/다리 일직선): 게이트 없이, 180도에서 벗어난 정도에
  선형으로 비례해 최대 25점까지 감점 (결과 25~50점 사이).
"""

from __future__ import annotations

import numpy as np

from .scorer import score_rep


def linear_penalty(deficit: float, width: float, max_penalty: float = 25.0) -> float:
    """
    deficit(목표 대비 부족한 정도)에 선형으로 비례해 감점한다.
    deficit=0이면 감점 0, deficit이 width 이상이면 감점이 max_penalty로 포화된다.
    부족한 매 1도가 똑같은 비중으로 깎이는, 이해하기 쉬운 감점 방식이다.
    """
    if deficit <= 0:
        return 0.0
    if deficit >= width:
        return max_penalty
    return float(max_penalty * deficit / width)


def gated_rule_score(
    depth_value: float,
    depth_gate: float,
    extension_value: float,
    extension_target: float,
    penalty_width: float,
    base_score: float = 50.0,
    max_penalty: float = 25.0,
) -> float:
    """
    "일정 깊이(depth_gate) 이상 굽혀야만 작동"하는 게이트형 규칙.
    depth_value가 depth_gate보다 크면(=충분히 안 굽혔으면) 0점.
    통과하면 base_score(50점)에서 시작해, extension_target 대비 extension_value가
    부족한 만큼 linear_penalty로 최대 max_penalty(25점)까지 감점한다.
    """
    if depth_value > depth_gate:
        return 0.0
    deficit = max(0.0, extension_target - extension_value)
    penalty = linear_penalty(deficit, penalty_width, max_penalty=max_penalty)
    return base_score - penalty


def ungated_rule_score(
    value: float,
    target: float,
    penalty_width: float,
    base_score: float = 50.0,
    max_penalty: float = 25.0,
) -> float:
    """
    게이트 없이, target(예: 180도, 일직선)에서 value가 벗어난 만큼 linear_penalty로
    최대 max_penalty(25점)까지 감점하는 규칙 ("깊이" 개념이 없는 몸통/다리 일직선 등에 사용).
    """
    deficit = max(0.0, target - value)
    penalty = linear_penalty(deficit, penalty_width, max_penalty=max_penalty)
    return base_score - penalty


def squat_rule_score(
    features: dict, depth_gate: float = 105.0, extension_target: float = 170.0, penalty_width: float = 40.0
) -> float:
    """squat: 무릎 깊이 게이트(105도) + 펴짐(170도 목표) 선형 감점(폭 40도)."""
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
    pushup 규칙 점수. elbow(팔꿈치 깊이 게이트 90도 + 펴짐 170도 목표, 최대 25점 선형 감점)가
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
    elbow_score = 50.0 - linear_penalty(elbow_deficit, elbow_penalty_width, max_penalty=25.0)

    hip_deficit = max(0.0, body_line_target - features["min_angle_hip"])
    hip_penalty = linear_penalty(hip_deficit, body_line_penalty_width, max_penalty=12.5)

    knee_deficit = max(0.0, body_line_target - features["min_angle_knee"])
    knee_penalty = linear_penalty(knee_deficit, body_line_penalty_width, max_penalty=12.5)

    return max(0.0, elbow_score - hip_penalty - knee_penalty)


RULE_SCORERS = {
    "squat": squat_rule_score,
    "pushup": pushup_rule_score,
}


# -----------------------------
# 규칙 채점 설명 (explain)
# -----------------------------

def explain_squat_rule(
    features: dict, depth_gate: float = 105.0, extension_target: float = 170.0, penalty_width: float = 40.0
) -> dict:
    """squat_rule_score의 계산 과정을 단계별 문장으로 풀어서 반환."""
    min_angle = features["min_angle_knee"]
    max_angle = features["max_angle_knee"]
    messages = []

    if min_angle > depth_gate:
        messages.append(
            f"무릎 최저각도가 {min_angle:.1f}도로 깊이 기준({depth_gate:.0f}도)을 통과하지 못해 "
            "규칙이 적용되지 않았습니다 (통계 점수만 사용됨)."
        )
        return {"gate_passed": False, "score": 0.0, "messages": messages}

    deficit = max(0.0, extension_target - max_angle)
    penalty = linear_penalty(deficit, penalty_width, max_penalty=25.0)
    score = 50.0 - penalty

    messages.append(f"무릎 최저각도 {min_angle:.1f}도로 깊이 기준({depth_gate:.0f}도) 통과 → 기본 50.0점")
    if deficit > 0:
        messages.append(
            f"무릎 최고각도가 {max_angle:.1f}도로 목표({extension_target:.0f}도)보다 "
            f"{deficit:.1f}도 부족 → {penalty:.1f}점 감점 (선형)"
        )
    else:
        messages.append(f"무릎 최고각도 {max_angle:.1f}도로 목표({extension_target:.0f}도) 이상 → 감점 없음")
    messages.append(f"최종 규칙 점수: {score:.1f}점")

    return {"gate_passed": True, "score": score, "deficit": deficit, "penalty": penalty, "messages": messages}


def explain_pushup_rule(
    features: dict,
    elbow_depth_gate: float = 90.0,
    elbow_extension_target: float = 170.0,
    elbow_penalty_width: float = 40.0,
    body_line_target: float = 180.0,
    body_line_penalty_width: float = 20.0,
) -> dict:
    """pushup_rule_score의 계산 과정(elbow 기본점 -> hip/knee 감점)을 단계별 문장으로 풀어서 반환."""
    min_angle_elbow = features["min_angle_elbow"]
    messages = []

    if min_angle_elbow > elbow_depth_gate:
        messages.append(
            f"팔꿈치 최저각도가 {min_angle_elbow:.1f}도로 깊이 기준({elbow_depth_gate:.0f}도)을 "
            "통과하지 못해 규칙이 적용되지 않았습니다 (통계 점수만 사용됨)."
        )
        return {"gate_passed": False, "score": 0.0, "messages": messages}

    elbow_deficit = max(0.0, elbow_extension_target - features["max_angle_elbow"])
    elbow_penalty = linear_penalty(elbow_deficit, elbow_penalty_width, max_penalty=25.0)
    elbow_score = 50.0 - elbow_penalty
    messages.append(f"팔꿈치 최저각도 {min_angle_elbow:.1f}도로 깊이 기준({elbow_depth_gate:.0f}도) 통과 → 기본 50.0점")
    if elbow_deficit > 0:
        messages.append(
            f"팔꿈치 최고각도가 목표({elbow_extension_target:.0f}도)보다 {elbow_deficit:.1f}도 부족 "
            f"→ {elbow_penalty:.1f}점 감점 (선형, elbow 소계 {elbow_score:.1f}점)"
        )

    hip_deficit = max(0.0, body_line_target - features["min_angle_hip"])
    hip_penalty = linear_penalty(hip_deficit, body_line_penalty_width, max_penalty=12.5)
    if hip_deficit > 0:
        messages.append(
            f"허리(엉덩이)가 일직선 기준({body_line_target:.0f}도)보다 {hip_deficit:.1f}도 처짐/솟음 "
            f"→ 추가 {hip_penalty:.1f}점 감점"
        )

    knee_deficit = max(0.0, body_line_target - features["min_angle_knee"])
    knee_penalty = linear_penalty(knee_deficit, body_line_penalty_width, max_penalty=12.5)
    if knee_deficit > 0:
        messages.append(
            f"다리가 일직선 기준({body_line_target:.0f}도)보다 {knee_deficit:.1f}도 굽음 "
            f"→ 추가 {knee_penalty:.1f}점 감점"
        )

    score = max(0.0, elbow_score - hip_penalty - knee_penalty)
    messages.append(f"최종 규칙 점수: {score:.1f}점")

    return {
        "gate_passed": True,
        "score": score,
        "elbow_penalty": elbow_penalty,
        "hip_penalty": hip_penalty,
        "knee_penalty": knee_penalty,
        "messages": messages,
    }


RULE_EXPLAINERS = {
    "squat": explain_squat_rule,
    "pushup": explain_pushup_rule,
}


def explain_rule_score(features: dict, exercise: str) -> dict:
    """
    exercise에 맞는 explain 함수를 찾아 규칙 채점 과정을 단계별로 풀어서 반환한다.
    정의된 규칙이 없는 exercise면 {"gate_passed": False, "score": 0.0, "messages": [...]}를 반환.
    """
    explain_fn = RULE_EXPLAINERS.get(exercise)
    if explain_fn is None:
        return {"gate_passed": False, "score": 0.0, "messages": [f"'{exercise}'에는 정의된 규칙이 없습니다."]}
    return explain_fn(features)


def score_rep_with_rules(
    features: dict, stats, exercise: str, decay_scale: float = 3.0, top_k_feedback: int = 2
) -> dict:
    """
    score_rep()(mahalanobis 기반)과 exercise별 규칙 점수를 max()로 합쳐서 채점.

    features: extract_rep_features()가 만든 dict (rep_features.csv의 한 행이어도 됨) —
        stats.feature_names에 대응하는 값들 + min_angle_knee 같은 원본 각도값들이
        모두 들어있어야 한다.

    Returns:
        score_rep()과 동일한 dict에 아래가 추가된 것:
        - "score_source": "rule" 또는 "mahalanobis" — 최종 점수가 어느 쪽에서 나왔는지
        - "rule_explanation": explain_rule_score()의 결과 (게이트 통과 여부, 단계별 감점
          내역, 사람이 읽을 수 있는 메시지 목록). exercise에 규칙이 없으면 빈 설명이 담김.
        규칙이 최종 점수를 결정했다면("score_source"=="rule") top_issues/outliers(mahalanobis
        기준 z-score)가 실제 감점 사유와 안 맞을 수 있으니, 이 경우엔 rule_explanation을
        우선 참고할 것.
    """
    feature_vector = np.array([features[name] for name in stats.feature_names])
    result = score_rep(feature_vector, stats, top_k_feedback=top_k_feedback, decay_scale=decay_scale)

    rule_explanation = explain_rule_score(features, exercise)
    result["rule_explanation"] = rule_explanation

    rule_fn = RULE_SCORERS.get(exercise)
    if rule_fn is None:
        result["score_source"] = "mahalanobis"
        return result

    rule_score = rule_explanation["score"]
    if rule_score > result["score"]:
        result["score"] = rule_score
        result["score_source"] = "rule"
    else:
        result["score_source"] = "mahalanobis"
    return result
