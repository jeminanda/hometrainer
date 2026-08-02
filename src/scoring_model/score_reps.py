"""
src/scoring_model/score_reps.py

학습된 ReferenceStats(models/{exercise}_reference.npz)를 불러와 rep을 채점한다.
mahalanobis(score_rep) 대신 rule_based.score_rep_with_rules를 써서, exercise별
규칙 기반 최소점수 보정(예: 깊은 스쿼트/팔굽혀펴기가 기준 평균과 달라서 낮게
나오는 것을 구제)이 항상 함께 적용된다.

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
from .rule_based import score_rep_with_rules
from .scorer import calibrate_decay_scale

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
    decay_scale: float = 3.0,
) -> list[dict]:
    """
    새 영상 1개(키포인트 .npy)를 템플릿으로 DTW 위상정규화 + 특징추출 + 채점까지 한 번에 처리.
    (build_dataset의 process_source_file_with_template + rule_based.score_rep_with_rules를
    이어붙인 것 — mahalanobis와 exercise별 규칙 점수를 max()로 합친 최종 점수가 나온다.)

    decay_scale: distance_to_score()의 감쇠 속도. 기본값(3.0)이 이 영상 특성상 너무
    가혹하다고 판단되면, scorer.calibrate_decay_scale()로 이 영상의 특정 rep 기준
    역산한 값을 넣어 다시 호출하면 된다 (calibrate_and_rescore_video 참고).

    Returns:
        rep마다 {video_id, exercise, rep_idx, ...특징들..., score, distance,
        feature_contributions, top_issues, score_source}
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
        score_result = score_rep_with_rules(row, stats, exercise, decay_scale=decay_scale)
        results.append({**row, **score_result})
    return results


def calibrate_and_rescore_video(
    keypoints_path: Path,
    exercise: str,
    template: np.ndarray,
    model_dir: Path,
    calibrate_rep_idx: int,
    calibrate_target_score: float,
    video_id: str | None = None,
    min_distance: int = 30,
    min_rep_frames: int = 30,
    inside_percentile: float = 95.0,
) -> tuple[list[dict], float]:
    """
    이 영상의 특정 rep(calibrate_rep_idx)이 원하는 점수(calibrate_target_score)가 나오도록
    decay_scale을 역산한 뒤, 그 값으로 영상 전체 rep을 다시 채점한다.

    주의: 이 역산은 mahalanobis distance 기준이다. 만약 calibrate_rep_idx로 고른 rep이
    이미 규칙 기반 점수(score_source="rule")로 결정된 상태라면, decay_scale을 바꿔도
    mahalanobis 쪽 점수만 바뀔 뿐 규칙 점수가 여전히 더 높으면 최종 점수는 안 바뀐다.

    사용 예: "무릎이 더 깊게 굽혀지는 이 영상의 3번째 rep은 90점은 돼야 할 것 같다" 싶을 때,
        results, decay_scale = calibrate_and_rescore_video(
            keypoints_path, exercise, template, model_dir,
            calibrate_rep_idx=3, calibrate_target_score=90.0,
        )
    처럼 부르면, 그 rep 기준으로 감쇠 속도를 맞춰서 영상 전체를 재채점해준다.
    (이 영상 하나만 다시 채점하는 것이라, 다른 영상/기준 모델 자체는 그대로다 —
    "이 영상을 볼 때는 이 정도로 관대하게 보고 싶다"는 일회성 조정에 가깝다.)

    Returns:
        (재채점된 results, 이번에 사용된 decay_scale)
    """
    # 1단계: 기본 decay_scale(3.0)로 우선 채점해서 기준 rep의 distance를 구한다.
    baseline_results = score_video(
        keypoints_path, exercise, template, model_dir,
        video_id=video_id, min_distance=min_distance, min_rep_frames=min_rep_frames,
    )
    target_rows = [r for r in baseline_results if r["rep_idx"] == calibrate_rep_idx]
    if not target_rows:
        available = [r["rep_idx"] for r in baseline_results]
        raise ValueError(f"rep_idx={calibrate_rep_idx}를 찾을 수 없습니다. 존재하는 rep_idx: {available}")

    if target_rows[0].get("score_source") == "rule":
        print(
            f"[정보] rep_idx={calibrate_rep_idx}는 이미 규칙 기반 점수({target_rows[0]['score']:.1f})로 "
            "결정돼 있습니다. decay_scale을 조정해도 규칙 점수보다 mahalanobis 점수가 높아지지 않으면 "
            "최종 점수는 안 바뀔 수 있습니다."
        )

    stats = load_reference(Path(model_dir) / f"{exercise}_reference.npz")
    new_decay_scale = calibrate_decay_scale(
        target_rows[0]["distance"], stats, calibrate_target_score, inside_percentile=inside_percentile
    )

    # 2단계: 역산한 decay_scale로 영상 전체를 다시 채점.
    results = score_video(
        keypoints_path, exercise, template, model_dir,
        video_id=video_id, min_distance=min_distance, min_rep_frames=min_rep_frames,
        decay_scale=new_decay_scale,
    )
    return results, new_decay_scale


def score_rep_sequence(rep_sequence: np.ndarray, exercise: str, model_dir: Path) -> dict:
    """
    (아직 rep_features.csv에 없는) 새 rep 시퀀스 1개 (target_length, J, C)를 채점.
    실시간/신규 영상 추론 경로에서 사용.

    Returns:
        {"score", "distance", "feature_contributions", "top_issues", "score_source"}
    """
    stats = load_reference(Path(model_dir) / f"{exercise}_reference.npz")
    features = extract_rep_features(rep_sequence, exercise)
    return score_rep_with_rules(features, stats, exercise)


def score_from_features(feature_row: pd.Series, stats) -> dict:
    """rep_features.csv에 이미 계산된 특징 행 1개로 채점 (재계산 없이 바로 사용)."""
    return score_rep_with_rules(feature_row, stats, stats.exercise)


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
            f"-> score={result['score']:.1f} (distance={result['distance']:.3f}, source={result['score_source']})"
        )
        print(f"   상위 이슈: {result['top_issues']}")


if __name__ == "__main__":
    main()
