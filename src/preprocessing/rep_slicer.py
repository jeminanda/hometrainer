import numpy as np
from scipy.signal import find_peaks

def slice_repetitions(time_series_angles: np.ndarray, min_distance: int = 15):
    """
    연속적인 각도 데이터(1D np.array)에서 반복 동작(Rep) 구간을 잘라냅니다.
    
    time_series_angles: 예) 프레임별 무릎/팔꿈치 각도 배열
    min_distance: minimum frame separation between reps
    """
    # 스쿼트/푸쉬업은 보통 각도가 최소(최대屈曲)가 되는 시점을 기준(Valley)으로 나눕니다.
    inverted_angles = -time_series_angles
    peaks, _ = find_peaks(inverted_angles, distance=min_distance, prominence=15)
    
    rep_segments = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i+1]
        rep_segments.append((start_idx, end_idx))
        
    return rep_segments