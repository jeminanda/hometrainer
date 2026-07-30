"""
src/scoring_model/reference_stats.py

"모범 사례" 특징 행렬(N_reps, num_features)로부터 평균(mean)과 공분산(covariance)을
학습해 Mahalanobis distance 계산에 필요한 통계치를 만들고 저장/로드한다.

가진 데이터가 전부 정상 수행 사례뿐이라 (부정 라벨 없음) 지도학습 분류기 대신
"정상 분포에서 얼마나 벗어났는가"를 재는 이 방식이 적합하다.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class ReferenceStats:
    exercise: str
    feature_names: list[str]
    mean: np.ndarray            # (num_features,)
    cov: np.ndarray              # (num_features, num_features), 정규화 적용됨
    inv_cov: np.ndarray           # cov의 역행렬 (Mahalanobis distance 계산용)
    reference_distances: np.ndarray  # (N_reps,) 정상 사례들 스스로의 거리 분포 (점수 보정용)


def fit_reference(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    exercise: str,
    reg_epsilon: float = 1e-6,
) -> ReferenceStats:
    """
    정상 사례 특징 행렬로부터 평균/공분산을 학습.

    Args:
        feature_matrix: (N_reps, num_features)
        feature_names: 컬럼 이름 (해석/디버깅용)
        exercise: "squat", "pushup" 등
        reg_epsilon: 공분산 행렬에 더할 작은 대각 성분 (표본 수가 적어 특이행렬이 되는 것 방지)

    Returns:
        ReferenceStats
    """
    N, num_features = feature_matrix.shape
    if N < num_features + 1:
        print(
            f"[경고] 표본 수({N})가 특징 수({num_features})보다 크게 여유롭지 않습니다. "
            "공분산 추정이 불안정할 수 있으니 reg_epsilon을 늘리거나 특징 수를 줄이는 것을 고려하세요."
        )

    mean = feature_matrix.mean(axis=0)
    cov = np.cov(feature_matrix, rowvar=False)
    cov = np.atleast_2d(cov)  # num_features==1인 극단적 경우 방지
    cov_reg = cov + np.eye(num_features) * reg_epsilon
    inv_cov = np.linalg.inv(cov_reg)

    # 정상 사례들 스스로의 거리 분포 (나중에 percentile 기반 점수 변환에 사용)
    diffs = feature_matrix - mean
    reference_distances = np.sqrt(np.einsum("ni,ij,nj->n", diffs, inv_cov, diffs))

    return ReferenceStats(
        exercise=exercise,
        feature_names=feature_names,
        mean=mean,
        cov=cov_reg,
        inv_cov=inv_cov,
        reference_distances=reference_distances,
    )


def save_reference(stats: ReferenceStats, path: Path) -> None:
    """ReferenceStats를 .npz 파일 하나로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        exercise=stats.exercise,
        feature_names=np.array(stats.feature_names),
        mean=stats.mean,
        cov=stats.cov,
        inv_cov=stats.inv_cov,
        reference_distances=stats.reference_distances,
    )


def load_reference(path: Path) -> ReferenceStats:
    """저장된 .npz 파일로부터 ReferenceStats 복원."""
    data = np.load(path, allow_pickle=False)
    return ReferenceStats(
        exercise=str(data["exercise"]),
        feature_names=list(data["feature_names"]),
        mean=data["mean"],
        cov=data["cov"],
        inv_cov=data["inv_cov"],
        reference_distances=data["reference_distances"],
    )
