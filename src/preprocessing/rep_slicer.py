import numpy as np
from scipy.signal import find_peaks

def slice_repetitions(time_series_angles: np.ndarray, min_distance: int = 15, min_rom: float = 20.0):
    """
    연속적인 각도 데이터(1D np.array)에서 반복 동작(Rep) 구간을 잘라냅니다.

    time_series_angles: 예) 프레임별 무릎/팔꿈치 각도 배열
    min_distance: minimum frame separation between reps
    min_rom: 극소점이 1개 이하일 때 "영상 전체를 rep 1개로 볼지" 판단하는 최소 가동범위(degree).
             이 값보다 각도 변화폭이 작으면 동작 자체가 없었다고 보고 빈 리스트를 반환한다.
    """
    # 스쿼트/푸쉬업은 보통 각도가 최소(최대屈曲)가 되는 시점을 기준(Valley)으로 나눕니다.
    inverted_angles = -time_series_angles
    peaks, _ = find_peaks(inverted_angles, distance=min_distance, prominence=15)

    if len(peaks) < 2:
        # valley-to-valley 방식은 극소점이 최소 2개 있어야 구간(=rep 1개)을 만들 수 있다.
        # 반복을 2회 이상 해야만 감지되는 구조라, "1회만 수행한 영상"은 극소점이 1개(또는 0개)뿐이라
        # 여기서 항상 빈 리스트가 나온다. 이런 영상은 대개 rep 1회짜리이므로,
        # 실제로 유의미한 가동범위(min_rom 이상)가 있었다면 영상 전체를 rep 1개로 간주한다.
        rom = float(np.max(time_series_angles) - np.min(time_series_angles))
        if rom >= min_rom and len(time_series_angles) > 1:
            return [(0, len(time_series_angles) - 1)]
        return []

    rep_segments = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i+1]
        rep_segments.append((start_idx, end_idx))
        
    return rep_segments