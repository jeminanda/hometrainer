"""
src/preprocessing/feature_extraction.py

preprocessing 파이프라인이 만든 정규화·위상정규화된 rep 시퀀스
(target_length, 33, C)에서 채점 모델(scoring_model)이 바로 쓸 수 있는
특징 벡터(관절 각도 min/max/ROM, 좌우 대칭성)를 추출한다.

(원래 scoring_model 쪽에 있었으나, README에 명시된 preprocessing 범위
"특징 추출: 관절 각도, ROM, 좌우 대칭성"에 해당하는 데이터 가공 로직이라
preprocessing으로 옮김. 특정 채점 방식(Mahalanobis 등)에 종속되지 않으므로
다른 모델을 시도해도 그대로 재사용 가능.)

한 rep 시퀀스는 이미 다음 처리가 끝난 상태다 (preprocessing 모듈 기준):
    - coordiante_normalization.normalize_landmarks : 힙 중심 정렬 + 몸통 스케일
    - phase_normalization.normalize_phase          : 고정 길이(target_length)로 리샘플링

주의: phase_normalization이 rep 길이를 전부 target_length로 강제로 맞추기 때문에
실제 수행 속도(rep이 몇 프레임 걸렸는지) 정보는 이 시퀀스 자체에는 남아있지 않다.
속도(템포) 특징이 필요하면 build_dataset 쪽에서 rep_features.csv에 duration_frames를
별도로 저장해 합쳐야 한다.
"""

from __future__ import annotations

import numpy as np

# angles.py의 EXERCISE_JOINTS/calculate_angle을 그대로 재사용 (중복 정의 방지)
from .angles import EXERCISE_JOINTS, calculate_angle


def compute_angle_series(rep_sequence: np.ndarray, joint_triplet: tuple[int, int, int], n_dims: int = 2) -> np.ndarray:
    """
    (target_length, J, C) rep 시퀀스에서 특정 관절 삼각형(A,B,C)의 프레임별 각도 시계열을 계산.

    n_dims=2 기본: BlazePoseExtractor 기본 출력이 (x, y, visibility)이므로
    visibility를 z로 오인하지 않도록 x, y만 사용.
    """
    a_idx, b_idx, c_idx = joint_triplet
    T = rep_sequence.shape[0]
    return np.array(
        [
            calculate_angle(
                rep_sequence[t, a_idx, :n_dims],
                rep_sequence[t, b_idx, :n_dims],
                rep_sequence[t, c_idx, :n_dims],
            )
            for t in range(T)
        ]
    )


def extract_rep_features(rep_sequence: np.ndarray, exercise: str, n_dims: int = 2) -> dict[str, float]:
    """
    rep 시퀀스 1개에서 exercise에 맞는 각도 기반 특징 딕셔너리를 추출.

    반환 예 (pushup): {
        "min_angle_elbow_left": ..., "max_angle_elbow_left": ..., "rom_elbow_left": ...,
        "min_angle_elbow_right": ..., "max_angle_elbow_right": ..., "rom_elbow_right": ...,
        "rom_symmetry_elbow": abs(rom_left - rom_right),
    }
    """
    if exercise not in EXERCISE_JOINTS:
        raise ValueError(f"'{exercise}'에 대한 각도 정의가 EXERCISE_JOINTS(angles.py)에 없습니다.")

    features: dict[str, float] = {}
    for joint_name, sides in EXERCISE_JOINTS[exercise].items():
        rom_by_side = {}
        for side, triplet in sides.items():
            series = compute_angle_series(rep_sequence, triplet, n_dims=n_dims)
            min_a, max_a = float(np.min(series)), float(np.max(series))
            rom = max_a - min_a
            features[f"min_angle_{joint_name}_{side}"] = min_a
            features[f"max_angle_{joint_name}_{side}"] = max_a
            features[f"rom_{joint_name}_{side}"] = rom
            rom_by_side[side] = rom

        if "left" in rom_by_side and "right" in rom_by_side:
            features[f"rom_symmetry_{joint_name}"] = abs(rom_by_side["left"] - rom_by_side["right"])

    return features


def build_feature_matrix(
    rep_sequences: np.ndarray, exercise: str, n_dims: int = 2
) -> tuple[np.ndarray, list[str]]:
    """
    여러 rep 시퀀스 (N, target_length, J, C)를 한 번에 특징 행렬 (N, num_features)로 변환.

    Returns:
        feature_matrix: (N, num_features)
        feature_names: 컬럼 순서에 대응하는 이름 리스트
    """
    all_features = [extract_rep_features(rep_sequences[i], exercise, n_dims=n_dims) for i in range(rep_sequences.shape[0])]
    feature_names = list(all_features[0].keys())
    feature_matrix = np.array([[f[name] for name in feature_names] for f in all_features])
    return feature_matrix, feature_names
