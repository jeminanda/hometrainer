"""
src/preprocessing/build_dataset.py

data/raw/ 의 pose_extraction 결과(키포인트 .npy)를 모두 순회하며
- coordiante_normalization.normalize_landmarks
- angles.select_reliable_side (좌우 중 visibility 높은 쪽 선택) + calculate_angle
- rep_slicer.slice_repetitions       (주의: (start_idx, end_idx) 인덱스 쌍을 반환함)
- phase_normalization.build_template + normalize_phase_dtw  (DTW 기반 위상 정규화)
- feature_extraction.extract_rep_features
만을 사용해 전처리를 완료하고, 결과를 data/processed/ 에 저장한다.

DTW 도입으로 2-패스 구조가 됐다 (기존 선형보간 방식은 파일 하나만 보고 그 자리에서
끝낼 수 있었지만, DTW는 "모범 사례 평균 궤적(템플릿)"이 먼저 있어야 하고, 그 템플릿은
전체 rep을 한 번 다 봐야 만들 수 있다):
    1패스: 모든 파일에서 rep 구간(가변 길이)까지만 뽑아 전부 모은다 (위상 정규화 전)
    -> exercise별로 템플릿(평균 각도 궤적) 생성
    2패스: 모아둔 각 rep을 해당 exercise의 템플릿에 DTW로 맞춰 위상 정규화 + 특징 추출

입력 파일 규칙 (기본):
    data/raw/{exercise}_{video_id}_keypoints.npy
    예) data/raw/squat_001_keypoints.npy
    규칙에 안 맞으면 --manifest 로 (video_id, keypoints_path, exercise) csv 지정 가능.

출력 (data/processed/):
    - rep_sequences.npy : (전체 rep 수, target_length, J, C) DTW로 위상 정규화된 rep 시퀀스
    - rep_features.csv  : rep_sequences.npy의 각 행이 어느 (video_id, exercise, rep_idx)인지 매핑 +
                          feature_extraction이 계산한 각도 특징(min/max/rom) + reliable_side 메타데이터
    - templates.npz     : exercise별 DTW 템플릿 (참고/디버깅용으로 함께 저장)
    - build_log.json    : 성공/실패 로그
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from .angles import EXERCISE_JOINTS, calculate_angle, select_reliable_side
from .coordiante_normalization import normalize_landmarks  # 파일명 오타(coordiante) 그대로 유지
from .feature_extraction import extract_rep_features
from .phase_normalization import build_template, normalize_phase_dtw
from .rep_slicer import slice_repetitions


# -----------------------------
# 입력 파일 탐색
# -----------------------------

@dataclasses.dataclass
class SourceFile:
    video_id: str
    exercise: str
    keypoints_path: Path


FILENAME_PATTERN = re.compile(r"^(?P<exercise>[a-zA-Z]+)_(?P<video_id>.+)_keypoints$")


def discover_source_files(raw_dir: Path) -> list[SourceFile]:
    sources = []
    for path in sorted(raw_dir.glob("*_keypoints.npy")):
        match = FILENAME_PATTERN.match(path.stem)
        if not match:
            print(f"[SKIP] 파일명 규칙 불일치, exercise 추론 불가: {path.name}")
            continue
        sources.append(
            SourceFile(
                video_id=match.group("video_id"),
                exercise=match.group("exercise").lower(),
                keypoints_path=path,
            )
        )
    return sources


def load_manifest(manifest_path: Path) -> list[SourceFile]:
    if pd is None:
        raise ImportError("manifest 사용에는 pandas가 필요합니다. `pip install pandas`")
    df = pd.read_csv(manifest_path)
    required_cols = {"video_id", "keypoints_path", "exercise"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv에 필요한 컬럼이 없습니다: {missing}")
    return [
        SourceFile(video_id=str(r.video_id), exercise=str(r.exercise).lower(), keypoints_path=Path(r.keypoints_path))
        for r in df.itertuples()
    ]


# -----------------------------
# 결측/저신뢰 프레임 처리
# (4개 모듈 어디에도 없지만, 없으면 angles.py의 arccos가 NaN을 뱉거나
#  visibility가 낮은 부정확한 좌표가 그대로 각도 계산에 들어가기 때문에 인라인 처리)
# -----------------------------

def _mask_low_visibility(keypoints: np.ndarray, visibility_channel: int = -1, threshold: float = 0.5) -> np.ndarray:
    """
    visibility가 threshold 미만인 관절의 (x, y)를 NaN으로 마스킹.
    visibility_channel=-1(마지막 채널) 기본값: (x,y,visibility) 3채널이든
    (x,y,z,visibility) 4채널이든 visibility는 항상 마지막 채널이므로 그대로 안전하게 동작한다.
    """
    out = keypoints.copy()
    low_conf = out[..., visibility_channel] < threshold
    out[..., 0][low_conf] = np.nan
    out[..., 1][low_conf] = np.nan
    return out


def _interpolate_missing_frames(keypoints: np.ndarray, fallback_raw: np.ndarray | None = None) -> np.ndarray:
    """
    프레임(시간) 축 기준 선형 보간으로 NaN(미검출/저신뢰) 좌표를 채운다.

    fallback_raw: 마스킹 전 원본 좌표. 특정 관절이 영상 전체에서 한 번도
    신뢰할 수 있었던 적이 없으면(예: 팔굽혀펴기 영상에서 무릎이 계속 화면 밖/저화질)
    시간축 보간으로는 채울 기준점 자체가 없어 NaN이 영구히 남는다.
    이 경우 NaN으로 두어 다운스트림(학습)을 깨뜨리기보다, 신뢰도는 낮아도
    원본에 실제로 찍힌 좌표값으로라도 채워 넣는다.
    """
    T, J, C = keypoints.shape
    out = keypoints.copy()
    t = np.arange(T)
    for j in range(J):
        for c in range(C):
            series = out[:, j, c]
            valid = ~np.isnan(series)
            if 0 < valid.sum() < T:
                out[:, j, c] = np.interp(t, t[valid], series[valid])

    if fallback_raw is not None:
        still_nan = np.isnan(out)
        if still_nan.any():
            out[still_nan] = fallback_raw[still_nan]

    return out


# -----------------------------
# 1패스: rep 구간(가변 길이)까지만 추출
# -----------------------------

@dataclasses.dataclass
class RawRepSegment:
    """위상 정규화 전, 가변 길이 그대로의 rep 하나."""
    video_id: str
    exercise: str
    rep_idx: int
    keypoints: np.ndarray       # (T_rep, J, C), 가변 길이
    angle_series: np.ndarray    # (T_rep,), rep_slicer가 쓴 것과 동일한 대표 각도 시계열


def extract_rep_segments(
    source: SourceFile,
    min_distance: int = 30,
    visibility_threshold: float = 0.5,
    min_rep_frames: int = 30,
) -> list[RawRepSegment]:
    """
    키포인트 파일 하나를 마스킹/보간 -> 정규화 -> 신뢰도 높은 쪽 선택 -> rep 슬라이싱까지
    처리해서, 위상 정규화 전(가변 길이) rep 목록을 반환한다.
    """
    keypoints_raw = np.load(source.keypoints_path)  # (T, J, C), C=3이면 (x, y, visibility)

    keypoints_masked = _mask_low_visibility(keypoints_raw, threshold=visibility_threshold)
    if np.isnan(keypoints_masked).any():
        keypoints_masked = _interpolate_missing_frames(keypoints_masked, fallback_raw=keypoints_raw)

    normalized = normalize_landmarks(keypoints_masked, n_spatial_dims=2)  # (T, J, C)

    # 측면 촬영에서는 한쪽 팔/다리가 구조적으로 가려지는 경우가 흔하므로(실측 확인됨),
    # 영상 전체 기준으로 더 신뢰도 높은 쪽을 골라 그 쪽 각도만으로 rep 경계를 탐지한다.
    joint_name = next(iter(EXERCISE_JOINTS[source.exercise]))
    reliable_side = select_reliable_side(normalized, source.exercise, joint_name)
    a_idx, b_idx, c_idx = EXERCISE_JOINTS[source.exercise][joint_name][reliable_side]

    T = normalized.shape[0]
    primary_angle_series = np.array(
        [calculate_angle(normalized[t, a_idx, :2], normalized[t, b_idx, :2], normalized[t, c_idx, :2]) for t in range(T)]
    )

    # slice_repetitions는 (start_idx, end_idx) '인덱스 쌍' 리스트를 반환한다 (슬라이싱된 배열이 아님!)
    rep_index_pairs = slice_repetitions(primary_angle_series, min_distance=min_distance)

    # 안전망: min_distance/min_prominence를 아무리 잘 튜닝해도, 최저점에서 잠깐 멈칫하는 동작
    # 습관이 있는 영상은 여전히 과도 분절될 수 있다. 진짜 rep이라면 최소 이 정도 프레임은
    # 지속돼야 한다는 길이 기준으로 비정상적으로 짧은 조각을 한 번 더 걸러낸다.
    rep_index_pairs = [(s, e) for s, e in rep_index_pairs if (e - s + 1) >= min_rep_frames]

    if len(rep_index_pairs) == 0:
        raise RuntimeError(
            f"{source.keypoints_path.name}: rep을 하나도 탐지하지 못했습니다 "
            f"('{joint_name}'({reliable_side}) 각도 기준 극소점 부족, 또는 전부 min_rep_frames 미만으로 걸러짐)."
        )

    return [
        RawRepSegment(
            video_id=source.video_id,
            exercise=source.exercise,
            rep_idx=rep_idx,
            keypoints=normalized[start:end + 1],
            angle_series=primary_angle_series[start:end + 1],
        )
        for rep_idx, (start, end) in enumerate(rep_index_pairs)
    ]


# -----------------------------
# 새 영상 1개 처리 (채점 시 사용 - 저장된 템플릿 재사용)
# -----------------------------

def process_source_file_with_template(
    source: SourceFile,
    template: np.ndarray,
    min_distance: int = 30,
    visibility_threshold: float = 0.5,
    min_rep_frames: int = 30,
) -> tuple[list[dict], np.ndarray]:
    """
    build_dataset()으로 미리 만들어 저장해둔 templates.npz의 템플릿을 그대로 받아,
    새 영상 1개를 DTW로 위상 정규화 + 특징 추출까지 처리한다. (채점 시점에는 기준
    데이터셋 전체로 템플릿을 다시 만들 필요 없이, 학습 때 쓴 템플릿을 그대로 재사용해야
    학습/채점 기준이 일치한다.)

    Returns:
        (index_rows, rep_sequences) - 기존 process_source_file()과 동일한 반환 형태
    """
    segments = extract_rep_segments(
        source, min_distance=min_distance, visibility_threshold=visibility_threshold, min_rep_frames=min_rep_frames
    )

    index_rows = []
    rep_sequences = []
    for seg in segments:
        rep_sequence = normalize_phase_dtw(seg.keypoints, seg.angle_series, template)
        row = {"video_id": seg.video_id, "exercise": seg.exercise, "rep_idx": seg.rep_idx}
        row.update(extract_rep_features(rep_sequence, seg.exercise))
        index_rows.append(row)
        rep_sequences.append(rep_sequence)

    return index_rows, np.stack(rep_sequences, axis=0)


# -----------------------------
# 전체 데이터셋 빌드 (2패스)
# -----------------------------

def build_dataset(
    raw_dir: Path,
    output_dir: Path,
    manifest_path: Optional[Path] = None,
    target_length: int = 100,
    min_distance: int = 30,
    min_rep_frames: int = 30,
) -> None:
    sources = load_manifest(manifest_path) if manifest_path else discover_source_files(raw_dir)
    print(f"처리 대상 파일 수: {len(sources)}")

    # ---- 1패스: 위상 정규화 전 rep 구간 전부 모으기 ----
    all_segments: list[RawRepSegment] = []
    log = {"success": [], "failed": []}

    for source in sources:
        try:
            segments = extract_rep_segments(
                source, min_distance=min_distance, min_rep_frames=min_rep_frames
            )
        except Exception as e:  # noqa: BLE001 - 한 파일 실패가 전체 빌드를 막지 않도록 함
            print(f"[FAIL] {source.keypoints_path.name}: {e}")
            log["failed"].append({"file": str(source.keypoints_path), "reason": str(e)})
            continue

        all_segments.extend(segments)
        log["success"].append({"file": str(source.keypoints_path), "num_reps": len(segments)})
        print(f"[OK] {source.keypoints_path.name}: rep {len(segments)}개 구간 추출 완료")

    if not all_segments:
        raise RuntimeError("성공적으로 처리된 파일이 없습니다. build_log.json을 확인하세요.")

    # ---- exercise별 DTW 템플릿 생성 (지금 모은 모범 사례들의 평균 궤적) ----
    exercises = sorted({seg.exercise for seg in all_segments})
    templates: dict[str, np.ndarray] = {}
    for exercise in exercises:
        angle_lists = [seg.angle_series for seg in all_segments if seg.exercise == exercise]
        templates[exercise] = build_template(angle_lists, target_length=target_length)
        print(f"[템플릿] '{exercise}': rep {len(angle_lists)}개 평균으로 템플릿 생성")

    # ---- 2패스: DTW 위상 정규화 + 특징 추출 ----
    all_index_rows: list[dict] = []
    all_sequences: list[np.ndarray] = []

    for seg in all_segments:
        template = templates[seg.exercise]
        rep_sequence = normalize_phase_dtw(seg.keypoints, seg.angle_series, template)

        row = {"video_id": seg.video_id, "exercise": seg.exercise, "rep_idx": seg.rep_idx}
        row.update(extract_rep_features(rep_sequence, seg.exercise))

        all_index_rows.append(row)
        all_sequences.append(rep_sequence)

    output_dir.mkdir(parents=True, exist_ok=True)

    rep_sequences_all = np.stack(all_sequences, axis=0)  # (N_reps_total, target_length, J, C)
    np.save(output_dir / "rep_sequences.npy", rep_sequences_all)

    np.savez(output_dir / "templates.npz", **templates)

    for i, row in enumerate(all_index_rows):
        row["seq_index"] = i

    if pd is not None:
        pd.DataFrame(all_index_rows).to_csv(output_dir / "rep_features.csv", index=False)
    else:
        with open(output_dir / "rep_features.json", "w", encoding="utf-8") as f:
            json.dump(all_index_rows, f, ensure_ascii=False, indent=2)

    with open(output_dir / "build_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(
        f"\n완료: rep {len(all_index_rows)}개 "
        f"(성공 파일 {len(log['success'])}개 / 실패 {len(log['failed'])}개)\n"
        f"rep_sequences shape: {rep_sequences_all.shape}\n"
        f"저장 위치: {output_dir}"
    )


# -----------------------------
# CLI 진입점
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="angles/coordiante_normalization/rep_slicer/phase_normalization(DTW)으로 data/processed 생성"
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--target-length", type=int, default=100)
    parser.add_argument("--min-distance", type=int, default=30, help="rep 사이 최소 프레임 간격")
    parser.add_argument("--min-rep-frames", type=int, default=30, help="이보다 짧은 rep은 과도 분절로 간주해 제외")
    args = parser.parse_args()

    build_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        target_length=args.target_length,
        min_distance=args.min_distance,
        min_rep_frames=args.min_rep_frames,
    )


if __name__ == "__main__":
    main()
