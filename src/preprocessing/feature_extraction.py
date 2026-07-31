"""
src/preprocessing/feature_extraction.py

preprocessing 파이프라인이 만든 정규화·위상정규화된 rep 시퀀스
(target_length, 33, C)에서 채점 모델(scoring_model)이 바로 쓸 수 있는
특징 벡터(관절 각도 min/max/ROM)를 추출한다.

측면 촬영에서는 카메라에 가까운 쪽/먼 쪽 팔·다리 중 한쪽이 구조적으로 가려져
visibility가 낮게 나오는 경우가 흔하다 (실측 사례: pushup 영상에서 한쪽 팔꿈치가
영상 전체 내내 visibility<0.5). 이런 상황에서 좌우를 모두 특징으로 쓰면 신뢰할 수
없는 쪽의 노이즈가 그대로 특징에 섞이고, 좌우 대칭성 특징은 아예 의미가 없어진다.
그래서 좌우를 각각 특징화하는 대신, `select_reliable_side`로 더 신뢰도 높은 쪽
하나만 골라 그 쪽의 각도만 특징으로 쓴다 (좌우 대칭성 특징은 제거).

(원래 scoring_model 쪽에 있었으나, README에 명시된 preprocessing 범위
"특징 추출: 관절 각도, ROM 등"에 해당하는 데이터 가공 로직이라 preprocessing으로 옮김.
특정 채점 방식(Mahalanobis 등)에 종속되지 않으므로 다른 모델을 시도해도 그대로 재사용 가능.)

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

# angles.py의 EXERCISE_JOINTS/calculate_angle/select_reliable_side를 그대로 재사용 (중복 정의 방지)
from .angles import EXERCISE_JOINTS, calculate_angle, select_reliable_side


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


def extract_rep_features(
    rep_sequence: np.ndarray, exercise: str, n_dims: int = 2, visibility_channel: int = -1
) -> dict[str, float]:
    """
    rep 시퀀스 1개에서 exercise에 맞는 각도 기반 특징 딕셔너리를 추출.
    좌우 중 이 rep에서 visibility가 더 높은 쪽만 사용한다 (select_reliable_side).

    반환 예 (pushup): {
        "min_angle_elbow": ..., "max_angle_elbow": ..., "rom_elbow": ...,
        "reliable_side_elbow": "left" 또는 "right" (특징 벡터에는 포함하지 않는 메타데이터 —
            build_dataset이 이 키를 감지해 rep_features.csv에 별도 컬럼으로만 남기고
            학습용 특징 행렬에서는 제외한다)
    }
    """
    if exercise not in EXERCISE_JOINTS:
        raise ValueError(f"'{exercise}'에 대한 각도 정의가 EXERCISE_JOINTS(angles.py)에 없습니다.")

    features: dict[str, float] = {}
    for joint_name, sides in EXERCISE_JOINTS[exercise].items():
        reliable_side = select_reliable_side(rep_sequence, exercise, joint_name, visibility_channel)
        series = compute_angle_series(rep_sequence, sides[reliable_side], n_dims=n_dims)
        min_a, max_a = float(np.min(series)), float(np.max(series))

        features[f"min_angle_{joint_name}"] = min_a
        features[f"max_angle_{joint_name}"] = max_a
        features[f"rom_{joint_name}"] = max_a - min_a
        features[f"reliable_side_{joint_name}"] = reliable_side  # 메타데이터 (문자열 - 특징 행렬 제외 대상)

    return features


def build_feature_matrix(
    rep_sequences: np.ndarray, exercise: str, n_dims: int = 2
) -> tuple[np.ndarray, list[str]]:
    """
    여러 rep 시퀀스 (N, target_length, J, C)를 한 번에 특징 행렬 (N, num_features)로 변환.
    `reliable_side_*` 메타데이터(문자열)는 학습용 숫자 행렬에서 제외한다.

    Returns:
        feature_matrix: (N, num_features)
        feature_names: 컬럼 순서에 대응하는 이름 리스트 (메타데이터 컬럼 제외)
    """
    all_features = [extract_rep_features(rep_sequences[i], exercise, n_dims=n_dims) for i in range(rep_sequences.shape[0])]
    feature_names = [k for k in all_features[0].keys() if not k.startswith("reliable_side_")]
    feature_matrix = np.array([[f[name] for name in feature_names] for f in all_features])
    return feature_matrix, feature_names
