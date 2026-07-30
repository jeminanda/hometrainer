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


def distance_to_score(distance: float, stats: ReferenceStats) -> float:
    """
    Mahalanobis distance를 0~100점으로 변환.

    정상 사례들 스스로의 거리 분포(stats.reference_distances) 안에서 이 거리가
    몇 백분위(percentile)에 해당하는지를 기준으로 점수를 매긴다.
    -> "정상 사례들끼리도 이 정도는 벌어져 있다"는 기준에 맞춰 점수가 보정되므로,
       임의의 배율(k) 하나로 고정하는 것보다 특징 개수/스케일 변화에 덜 민감하다.
    """
    percentile = float(np.mean(stats.reference_distances <= distance)) * 100  # 0~100
    score = 100.0 - percentile  # 정상 분포보다 많이 벗어날수록(=percentile 높을수록) 감점
    return max(0.0, min(100.0, score))


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
