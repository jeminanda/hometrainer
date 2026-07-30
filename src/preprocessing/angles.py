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

# 운동별 주요 관절 정의: {exercise: {joint_name: {"left": (A,B,C), "right": (A,B,C)}}}
# extract_exercise_angles와 feature_extraction.py가 이 정의 하나만 공유해서 사용한다
# (예전엔 angles.py의 if/elif와 feature_extraction.py의 딕셔너리에 같은 내용이 중복돼 있었음).
EXERCISE_JOINTS = {
    "squat": {
        "knee": {
            "left": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            "right": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        },
    },
    "pushup": {
        "elbow": {
            "left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        },
    },
}


def extract_exercise_angles(row_data: np.ndarray, exercise_type: str = "squat") -> dict:
    """
    1개 프레임의 랜드마크 데이터를 받아 주요 운동 관절 각도를 추출합니다.

    row_data: (33, C) ndarray. BlazePoseExtractor 기본 출력은 (x, y, visibility)이므로
              앞 2개 채널(x, y)만 사용한다 — visibility(3번째 채널)를 z로 오인해
              각도가 틀어지는 것을 방지하기 위함. 실제로 z를 추출했다면 n_dims=3으로 바꿀 것.

    반환 키는 "{joint_name}_{side}" 형식 (예: "knee_left", "elbow_right").
    """
    n_dims = 2  # (x, y, visibility) 기준. z를 실제로 갖고 있다면 3으로 변경.

    if exercise_type not in EXERCISE_JOINTS:
        raise ValueError(f"정의되지 않은 exercise_type: {exercise_type}")

    angles = {}
    for joint_name, sides in EXERCISE_JOINTS[exercise_type].items():
        for side, (a_idx, b_idx, c_idx) in sides.items():
            angles[f"{joint_name}_{side}"] = calculate_angle(
                row_data[a_idx, :n_dims], row_data[b_idx, :n_dims], row_data[c_idx, :n_dims]
            )

    return angles