"""
src/scoring_model/train.py

preprocessing이 만든 data/processed/rep_features.csv(각도 특징이 이미 계산되어 있음)를
읽어, exercise별로 정상 사례 통계(ReferenceStats)를 학습해 저장한다.

(예전엔 여기서 rep_sequences.npy를 다시 읽어 특징을 재계산했지만, feature_extraction이
preprocessing으로 옮겨가면서 build_dataset 단계에서 이미 계산해둔 rep_features.csv를
그대로 쓰도록 변경 — 중복 계산 제거)

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


def train_for_exercise(processed_dir: Path, exercise: str, output_dir: Path) -> None:
    rep_features_df = pd.read_csv(processed_dir / "rep_features.csv")

    subset = rep_features_df[rep_features_df["exercise"] == exercise]
    if len(subset) == 0:
        raise ValueError(f"rep_features.csv에 exercise='{exercise}'인 rep이 없습니다.")

    feature_names = [c for c in subset.columns if c not in META_COLUMNS]
    feature_matrix = subset[feature_names].to_numpy()
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
