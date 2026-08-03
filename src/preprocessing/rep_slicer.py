import numpy as np
from scipy.signal import find_peaks

def slice_repetitions(
    time_series_angles: np.ndarray, min_distance: int = 30, min_prominence: float = 15.0, min_rom: float = 20.0
):
    """
    연속적인 각도 데이터(1D np.array)에서 극대점(Peak) 및 영상 시작/끝 지점을 포함하여
    반복 동작(Rep) 구간을 잘라냅니다.

    time_series_angles: 프레임별 각도 배열
    min_distance: Peak 간의 최소 프레임 간격
    min_prominence: find_peaks prominence 임계값
    min_rom: 의미 있는 동작으로 인정하기 위한 최소 가동범위(degree)
    """
    n_frames = len(time_series_angles)
    if n_frames < 2:
        return []

    # 1. 내부 극대점(Peak) 탐색
    raw_peaks, _ = find_peaks(time_series_angles, distance=min_distance, prominence=min_prominence)
    peaks = list(raw_peaks)

    # 2. 영상 시작(0) 및 끝(n_frames - 1) 경계 지점 보완
    # 기존 Peak와 너무 가까운 경계값은 중복 방지를 위해 제외 (min_distance / 2 기준)
    min_boundary_gap = min_distance // 2

    if len(peaks) == 0:
        boundaries = [0, n_frames - 1]
    else:
        boundaries = []
        
        # 시작 지점(0) 추가 여부 판단
        if peaks[0] >= min_boundary_gap:
            boundaries.append(0)
            
        boundaries.extend(peaks)
        
        # 끝 지점(n_frames - 1) 추가 여부 판단
        if (n_frames - 1) - peaks[-1] >= min_boundary_gap:
            boundaries.append(n_frames - 1)

    # 3. 경계 지점들 간의 구간 중, 가동범위(ROM) 조건(min_rom)을 만족하는 구간만 슬라이싱
    rep_segments = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]

        # 해당 구간 내의 최대 각도와 최소 각도 차이(ROM) 계산
        segment_angles = time_series_angles[start_idx : end_idx + 1]
        rom = float(np.max(segment_angles) - np.min(segment_angles))

        # 가동범위가 min_rom 이상일 때만 유효한 Rep으로 간주 (동작 없이 가만히 멈춰있는 시작/끝 구간 필터링)
        if rom >= min_rom:
            rep_segments.append((start_idx, end_idx))

    return rep_segments