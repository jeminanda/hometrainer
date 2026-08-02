"""
src/scoring_model/score_reps.py

학습된 ReferenceStats(models/{exercise}_reference.npz)를 불러와 rep을 채점한다.

두 가지 사용 시나리오:
1) 아직 채점 안 해본 새 영상의 rep 시퀀스 하나를 채점 -> score_rep_sequence()
   (전처리 파이프라인을 새로 거친 rep_sequence를 그 자리에서 특징 추출해 채점)
2) data/processed에 이미 있는 rep들을 다시 채점(데모/검증용) -> CLI(main())
   (build_dataset이 이미 rep_features.csv에 특징을 계산해뒀으므로 재계산 없이 바로 사용)

CLI 사용 예:
    python -m src.scoring_model.score_reps \\
        --processed-dir data/processed --exercise pushup --model-dir models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..preprocessing.build_dataset import SourceFile, process_source_file_with_template
from ..preprocessing.feature_extraction import extract_rep_features
from .reference_stats import load_reference
from .scorer import score_rep

META_COLUMNS = {"video_id", "exercise", "rep_idx", "seq_index"}


def load_template(processed_dir: Path, exercise: str) -> np.ndarray:
    """
    build_dataset()이 저장해둔 templates.npz에서 exercise의 DTW 템플릿을 불러온다.
    채점 시에는 반드시 학습 때 쓴 것과 동일한 템플릿을 재사용해야 한다 — 새로
    템플릿을 만들면 학습/채점 기준이 어긋난다.
    """
    with np.load(Path(processed_dir) / "templates.npz") as data:
        if exercise not in data:
            raise KeyError(f"templates.npz에 '{exercise}' 템플릿이 없습니다. 사용 가능: {list(data.keys())}")
        return data[exercise].copy()


def score_video(
    keypoints_path: Path,
    exercise: str,
    template: np.ndarray,
    model_dir: Path,
    video_id: str | None = None,
    min_distance: int = 30,
    min_rep_frames: int = 30,
) -> list[dict]:
    """
    새 영상 1개(키포인트 .npy)를 템플릿으로 DTW 위상정규화 + 특징추출 + 채점까지 한 번에 처리.
    (build_dataset의 process_source_file_with_template + scorer.score_rep을 이어붙인 것)

    Returns:
        rep마다 {video_id, exercise, rep_idx, ...특징들..., score, distance, feature_contributions, top_issues}
    """
    keypoints_path = Path(keypoints_path)
    source = SourceFile(
        video_id=video_id or keypoints_path.stem,
        exercise=exercise,
        keypoints_path=keypoints_path,
    )
    index_rows, _ = process_source_file_with_template(
        source, template, min_distance=min_distance, min_rep_frames=min_rep_frames
    )

    stats = load_reference(Path(model_dir) / f"{exercise}_reference.npz")

    results = []
    for row in index_rows:
        feature_vector = np.array([row[name] for name in stats.feature_names])
        score_result = score_rep(feature_vector, stats)
        results.append({**row, **score_result})
    return results


def score_rep_sequence(rep_sequence: np.ndarray, exercise: str, model_dir: Path) -> dict:
    """
    (아직 rep_features.csv에 없는) 새 rep 시퀀스 1개 (target_length, J, C)를 채점.
    실시간/신규 영상 추론 경로에서 사용.

    Returns:
        {"score", "distance", "feature_contributions", "top_issues"}
    """
    stats = load_reference(Path(model_dir) / f"{exercise}_reference.npz")
    features = extract_rep_features(rep_sequence, exercise)
    feature_vector = np.array([features[name] for name in stats.feature_names])
    return score_rep(feature_vector, stats)


def score_from_features(feature_row: pd.Series, stats) -> dict:
    """rep_features.csv에 이미 계산된 특징 행 1개로 채점 (재계산 없이 바로 사용)."""
    feature_vector = np.array([feature_row[name] for name in stats.feature_names])
    return score_rep(feature_vector, stats)


def main():
    parser = argparse.ArgumentParser(description="rep_features.csv에 이미 계산된 rep들을 저장된 ReferenceStats로 채점")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--exercise", type=str, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--limit", type=int, default=5, help="출력할 rep 개수 (앞에서부터)")
    args = parser.parse_args()

    rep_features_df = pd.read_csv(args.processed_dir / "rep_features.csv")
    stats = load_reference(Path(args.model_dir) / f"{args.exercise}_reference.npz")

    subset = rep_features_df[rep_features_df["exercise"] == args.exercise].head(args.limit)

    for _, row in subset.iterrows():
        result = score_from_features(row, stats)
        print(
            f"video={row['video_id']} rep={row['rep_idx']} "
            f"-> score={result['score']:.1f} (distance={result['distance']:.3f})"
        )
        print(f"   상위 이슈: {result['top_issues']}")


if __name__ == "__main__":
    main()
