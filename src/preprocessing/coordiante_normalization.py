import numpy as np

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    3D 랜드마크 좌표를 골반 중앙점 기준으로 정규화합니다.
    
    landmarks: shape (33, 3) 또는 (N_frames, 33, 3) 
               (x, y, z 좌표를 포함하는 MediaPipe Pose 랜드마크)
               
    MediaPipe Pose Landmark Index:
    - 11: Left Shoulder, 12: Right Shoulder
    - 23: Left Hip, 24: Right Hip
    """
    is_single_frame = (landmarks.ndim == 2)
    if is_single_frame:
        landmarks = np.expand_dims(landmarks, axis=0) # (1, 33, 3) 형태로 변환

    # 1. 골반 중앙점 (Mid-Hip) 계산 (Index 23, 24)
    left_hip = landmarks[:, 23, :]
    right_hip = landmarks[:, 24, :]
    mid_hip = (left_hip + right_hip) / 2.0  # shape: (N_frames, 3)

    # 2. 중심 이동 (Mid-Hip을 원점 (0,0,0)으로 설정)
    # broadcasting을 활용하여 모든 랜드마크에서 mid_hip 좌표를 뺍니다.
    centered_landmarks = landmarks - np.expand_dims(mid_hip, axis=1)

    # 3. 신체 스케일 정규화 (Scale Invariance)
    # 양 어깨(Index 11, 12) 사이의 거리를 계산하여 스케일 기준값으로 사용
    left_shoulder = landmarks[:, 11, :]
    right_shoulder = landmarks[:, 12, :]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True) # shape: (N_frames, 1)

    # 0으로 나누어지는 것(Division by zero) 방지 (eps = 1e-6)
    shoulder_width = np.maximum(shoulder_width, 1e-6)

    # 4. 좌표를 어깨 너비로 나누어 스케일 정규화
    normalized_landmarks = centered_landmarks / np.expand_dims(shoulder_width, axis=-1)

    if is_single_frame:
        return normalized_landmarks[0] # 원본 형태 (33, 3)로 복원
        
    return normalized_landmarks