import numpy as np

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    세 점 a, b, c 사이의 각도(b가 꼭짓점)를 3차원 벡터 내적으로 계산합니다.
    a, b, c: np.array([x, y, z])
    """
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return float(np.degrees(angle))

# MediaPipe BlazePose 33 랜드마크 인덱스 (표준)
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def extract_exercise_angles(row_data: np.ndarray, exercise_type: str = "squat") -> dict:
    """
    1개 프레임의 랜드마크 데이터를 받아 주요 운동 관절 각도를 추출합니다.

    row_data: (33, C) ndarray. BlazePoseExtractor 기본 출력은 (x, y, visibility)이므로
              앞 2개 채널(x, y)만 사용한다 — visibility(3번째 채널)를 z로 오인해
              각도가 틀어지는 것을 방지하기 위함. 실제로 z를 추출했다면 n_dims=3으로 바꿀 것.
    """
    n_dims = 2  # (x, y, visibility) 기준. z를 실제로 갖고 있다면 3으로 변경.

    angles = {}
    if exercise_type == "squat":
        angles["left_knee"] = calculate_angle(
            row_data[LEFT_HIP, :n_dims], row_data[LEFT_KNEE, :n_dims], row_data[LEFT_ANKLE, :n_dims]
        )
        angles["right_knee"] = calculate_angle(
            row_data[RIGHT_HIP, :n_dims], row_data[RIGHT_KNEE, :n_dims], row_data[RIGHT_ANKLE, :n_dims]
        )
    elif exercise_type == "pushup":
        angles["left_elbow"] = calculate_angle(
            row_data[LEFT_SHOULDER, :n_dims], row_data[LEFT_ELBOW, :n_dims], row_data[LEFT_WRIST, :n_dims]
        )
        angles["right_elbow"] = calculate_angle(
            row_data[RIGHT_SHOULDER, :n_dims], row_data[RIGHT_ELBOW, :n_dims], row_data[RIGHT_WRIST, :n_dims]
        )
    else:
        raise ValueError(f"정의되지 않은 exercise_type: {exercise_type}")

    return angles