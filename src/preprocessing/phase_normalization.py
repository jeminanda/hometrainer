import numpy as np
from scipy.interpolate import interp1d

def normalize_phase(rep_data: np.ndarray, target_length: int = 100) -> np.ndarray:
    """
    가변 길이의 1회 Rep 데이터를 고정 길이(예: 100 프레임)로 선형 보간 정규화합니다.
    
    rep_data: shape (N_frames, num_features)
    """
    current_length = rep_data.shape[0]
    if current_length < 2:
        return np.zeros((target_length, rep_data.shape[1]))
        
    original_timeline = np.linspace(0, 1, current_length)
    target_timeline = np.linspace(0, 1, target_length)
    
    interpolator = interp1d(original_timeline, rep_data, axis=0, kind='linear')
    normalized_data = interpolator(target_timeline)
    
    return normalized_data