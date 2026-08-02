"""
src/scoring_model/train.py

preprocessing이 만든 data/processed/rep_features.csv(각도 특징이 이미 계산되어 있음)를
읽어, exercise별로 정상 사례 통계(ReferenceStats)를 학습해 저장한다.

(예전엔 여기서 rep_sequences.npy를 다시 읽어 특징을 재계산했지만, feature_extraction이
preprocessing으로 옮겨가면서 build_dataset 단계에서 이미 계산해둔 rep_features.csv를
그대로 쓰도록 변경 — 중복 계산 제거)

(영상별 중앙값 대비 편차 특징(normalization.py)을 검토했었지만, 실제 데이터로 원인을
까본 결과 distance가 뭉치는 원인이 "사람 간 체형 차이"가 아니라 카메라 각도 문제/템포
차이로 확인되어 롤백함 — 프로젝트에는 적용하지 않기로 함. 필요성이 다시 제기되면
normalization.py의 add_within_group_deviation을 그때 다시 연결하면 된다.)

사용 예:
    python -m src.scoring_model.train \\
        --processed-dir data/processed --exercise pushup --output-dir models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .reference_stats import fit_reference, save_reference

# rep_features.csv의 메타데이터 컬럼 (특징 컬럼이 아닌 것들)
META_COLUMNS = {"video_id", "exercise", "rep_idx", "seq_index"}


def is_feature_column(col: str) -> bool:
    """META_COLUMNS도 아니고 reliable_side_*(문자열 메타데이터)도 아닌 컬럼만 특징으로 취급."""
    return col not in META_COLUMNS and not col.startswith("reliable_side_")


def train_for_exercise(processed_dir: Path, exercise: str, output_dir: Path) -> None:
    rep_features_df = pd.read_csv(processed_dir / "rep_features.csv")

    subset = rep_features_df[rep_features_df["exercise"] == exercise].reset_index(drop=True)
    if len(subset) == 0:
        raise ValueError(f"rep_features.csv에 exercise='{exercise}'인 rep이 없습니다.")

    candidate_names = [c for c in subset.columns if is_feature_column(c)]

    # rep_features.csv에 여러 운동이 함께 저장돼 있으면, build_dataset이 운동별로
    # 서로 다른 특징 컬럼(예: squat=*_knee, pushup=*_elbow)을 만들기 때문에
    # 지금 exercise가 쓰지 않는 다른 운동의 컬럼은 이 subset 안에서 전부 NaN으로 채워진다.
    # 그런 컬럼이 특징 행렬에 섞여 들어가면 mean/cov 계산 자체가 NaN으로 오염되므로 제외한다.
    all_nan_columns = [c for c in candidate_names if subset[c].isna().all()]
    if all_nan_columns:
        print(
            f"[정보] '{exercise}'에서 전부 NaN인 컬럼 {all_nan_columns}을(를) 특징에서 제외합니다 "
            "(다른 운동 전용 컬럼이 rep_features.csv에 함께 저장돼 있는 경우 정상입니다)."
        )
    feature_names = [c for c in candidate_names if c not in all_nan_columns]

    feature_matrix = subset[feature_names].to_numpy()

    if pd.isna(feature_matrix).any():
        raise ValueError(
            f"'{exercise}' 특징 행렬에 일부 NaN이 남아있습니다 (전부 NaN인 컬럼이 아니라 "
            "일부 행만 NaN). 결측 처리가 안 된 rep이 섞여 있을 수 있으니 rep_features.csv를 확인하세요."
        )

    print(f"'{exercise}' rep {len(subset)}개로 학습 시작, 특징 벡터 shape: {feature_matrix.shape}")
    print(f"특징: {feature_names}")

    stats = fit_reference(feature_matrix, feature_names, exercise=exercise)

    output_path = Path(output_dir) / f"{exercise}_reference.npz"
    save_reference(stats, output_path)

    print(
        f"기준 거리 분포 (percentile 스코어 보정용): "
        f"min={stats.reference_distances.min():.3f}, "
        f"median={stats.reference_distances.mean():.3f}, "
        f"max={stats.reference_distances.max():.3f}"
    )
    print(f"저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="exercise별 정상 사례 통계(ReferenceStats) 학습")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--exercise", type=str, required=True, help="squat, pushup 등")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    train_for_exercise(args.processed_dir, args.exercise, args.output_dir)


if __name__ == "__main__":
    main()
