"""
src/scoring_model/scorer.py

특징 벡터와 ReferenceStats(정상 사례 통계)를 받아 Mahalanobis distance를 계산하고,
이를 0~100점 스코어로 변환한다. 어떤 관절/각도가 감점에 가장 크게 기여했는지도
함께 계산해 "무릎 가동범위가 부족합니다" 같은 피드백에 쓸 수 있게 한다.
"""

from __future__ import annotations

import numpy as np

from .reference_stats import ReferenceStats


def mahalanobis_distance(feature_vector: np.ndarray, stats: ReferenceStats) -> float:
    """정상 사례 분포 기준으로 feature_vector까지의 Mahalanobis distance."""
    diff = feature_vector - stats.mean
    return float(np.sqrt(diff @ stats.inv_cov @ diff))


def distance_to_score(
    distance: float,
    stats: ReferenceStats,
    method: str = "threshold",
    inside_percentile: float = 95.0,
    decay_scale: float = 3.0,
) -> float:
    """
    Mahalanobis distance를 0~100점으로 변환.

    method="percentile": 기준 집단(stats.reference_distances) 안에서의 상대 순위로 점수를 매긴다.
        문제는, 학습에 쓴 데이터가 전부 "모범 사례"인데도 이 방식은 그 모범 사례들끼리
        상대 순위를 매겨버려서, 완전히 정상적인 폼이어도 그저 기준 집단의 중앙값 근처에
        있다는 이유만으로 50점 근처가 나온다 (교과서적인 자세인데도 만점을 못 받는 셈).
        기준 집단의 최댓값을 넘어서면 무조건 0점으로 포화되는 문제도 있다.

    method="threshold"(기본): 학습 데이터가 전부 정상 사례라는 전제를 그대로 살려서,
        "기준 집단이 보여준 자연스러운 변동 범위 안"이면 그냥 100점을 준다. 그 범위를
        벗어나야만 감점이 시작되고, 벗어난 정도(스프레드 단위)에 비례해 완만하게 줄어든다.
        경계는 max(reference_distances)가 아니라 inside_percentile(기본 95%)로 잡는다 —
        기준 데이터 안에 노이즈 섞인 이상치가 하나라도 있으면(실제로 겪었던 rep 과다분절
        사례처럼) max를 그대로 쓸 경우 그 이상치 하나가 "만점 구간"의 크기를 왜곡시키기
        때문에, 상위 몇 % 정도는 이상치로 보고 무시하는 편이 더 안정적이다.

        decay_scale: 경계를 넘은 뒤 감쇠 속도. `score = 100 * exp(-초과분 / decay_scale)`.
        값이 클수록 완만해진다. decay_scale=1(첫 시도)이었을 때는 경계를 스프레드 1배만
        넘어도 100 -> 37점으로 너무 가파르게 떨어져서, 기본값을 3으로 완만하게 바꿨다
        (경계+1스프레드에서 100->72점, +3스프레드에서 37점 정도로 덜 급격하게 감).
    """
    if method == "percentile":
        percentile = float(np.mean(stats.reference_distances <= distance)) * 100
        return float(np.clip(100.0 - percentile, 0.0, 100.0))

    if method != "threshold":
        raise ValueError(f"알 수 없는 method: {method} (percentile/threshold 중 하나)")

    threshold = float(np.percentile(stats.reference_distances, inside_percentile))
    if distance <= threshold:
        return 100.0

    spread = float(np.std(stats.reference_distances)) + 1e-6
    excess = (distance - threshold) / spread  # 경계를 몇 스프레드만큼 넘었는지
    return float(100.0 * np.exp(-excess / decay_scale))


def per_feature_contribution(feature_vector: np.ndarray, stats: ReferenceStats) -> dict[str, float]:
    """
    각 특징이 평균에서 얼마나 벗어났는지(표준편차 단위 z-score)를 계산.
    공분산 간 상호작용까지는 반영하지 않지만, "어느 각도가 문제인지" 피드백용으로 충분히 유용하다.

    Returns:
        {feature_name: z-score} — 절댓값이 클수록 그 특징이 크게 벗어났다는 뜻
    """
    std = np.sqrt(np.diag(stats.cov))
    std = np.where(std < 1e-8, 1e-8, std)
    z_scores = (feature_vector - stats.mean) / std
    return dict(zip(stats.feature_names, z_scores.tolist()))


def score_rep(feature_vector: np.ndarray, stats: ReferenceStats, top_k_feedback: int = 2) -> dict:
    """
    rep 1개를 채점하는 엔드투엔드 함수.

    Returns:
        {
            "score": 0~100,
            "distance": Mahalanobis distance,
            "feature_contributions": {feature_name: z-score, ...},
            "top_issues": [(feature_name, z-score), ...] 절댓값 기준 상위 top_k_feedback개
        }
    """
    distance = mahalanobis_distance(feature_vector, stats)
    score = distance_to_score(distance, stats)
    contributions = per_feature_contribution(feature_vector, stats)

    top_issues = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k_feedback]

    return {
        "score": score,
        "distance": distance,
        "feature_contributions": contributions,
        "top_issues": top_issues,
    }
