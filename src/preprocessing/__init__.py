from .angles import calculate_angle, extract_exercise_angles, select_reliable_side, EXERCISE_JOINTS
from .rep_slicer import slice_repetitions
from .phase_normalization import normalize_phase, build_template, normalize_phase_dtw
from .coordiante_normalization import normalize_landmarks
from .feature_extraction import extract_rep_features, build_feature_matrix, compute_angle_series
from .visualizer import animate_skeleton_2d  # 추가
from .build_dataset import (
    process_source_file_with_template,
    extract_rep_segments,
    discover_source_files,
    load_manifest,
    _interpolate_missing_frames,
    build_dataset,
)

__all__ = [
    "calculate_angle",
    "extract_exercise_angles",
    "select_reliable_side",  # 좌우 중 visibility 높은 쪽 선택
    "EXERCISE_JOINTS",
    "slice_repetitions",
    "normalize_phase",
    "build_template",              # 추가 - DTW 템플릿 생성
    "normalize_phase_dtw",         # 추가 - DTW 기반 위상 정규화
    "normalize_landmarks",
    "extract_rep_features",
    "build_feature_matrix",
    "compute_angle_series",
    "animate_skeleton_2d",
    "process_source_file_with_template",  # process_source_file 대체 (템플릿 인자 필요)
    "extract_rep_segments",               # 추가 - 위상정규화 전 rep 구간 추출
    "discover_source_files",
    "load_manifest",
    "_interpolate_missing_frames",
    "build_dataset",
]
