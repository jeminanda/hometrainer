import numpy as np


def normalize_landmarks(landmarks: np.ndarray, n_spatial_dims: int = 2) -> np.ndarray:
    """3D(또는 2D) 랜드마크를 골반 중앙점 기준으로 정규화.

    landmarks: (33, C) 또는 (N_frames, 33, C)
    n_spatial_dims: 실제 좌표 채널 수. z가 없는 (x, y, visibility) 데이터라면
                    반드시 2로 지정해야 visibility가 z로 오인되지 않는다.
    """
    is_single_frame = landmarks.ndim == 2
    if is_single_frame:
        landmarks = np.expand_dims(landmarks, axis=0)

    N_frames, J, C = landmarks.shape
    if C < n_spatial_dims:
        raise ValueError(f"n_spatial_dims={n_spatial_dims}인데 채널 수 C={C}로는 부족합니다.")

    sl = slice(0, n_spatial_dims)

    # MediaPipe 표준: 11=LEFT_SHOULDER, 12=RIGHT_SHOULDER, 23=LEFT_HIP, 24=RIGHT_HIP
    left_hip, right_hip = landmarks[:, 23, sl], landmarks[:, 24, sl]
    left_shoulder, right_shoulder = landmarks[:, 11, sl], landmarks[:, 12, sl]

    mid_hip = (left_hip + right_hip) / 2.0
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0

    centered = landmarks.copy()
    centered[:, :, sl] = landmarks[:, :, sl] - mid_hip[:, np.newaxis, :]

    # 어깨너비 대신 '몸통 길이'(어깨중심-힙중심) 사용: 더 길고 회전에 덜 민감함
    torso_length = np.linalg.norm(mid_shoulder - mid_hip, axis=-1)  # (N_frames,)

    # 프레임별로 따로 나누지 않고, 시퀀스 전체의 대표값(중앙값) 하나로 고정
    # -> 애니메이션 재생 중 사람 크기가 프레임마다 출렁이는 것을 방지
    reference_scale = np.median(torso_length)
    reference_scale = max(reference_scale, 1e-3)  # 하한선도 더 현실적인 값으로 상향

    centered[:, :, sl] = centered[:, :, sl] / reference_scale

    return centered[0] if is_single_frame else centered