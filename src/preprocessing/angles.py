from __future__ import annotations

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
        # elbow가 항상 첫 번째로 남아있어야 함: build_dataset.py의 rep 경계 탐지가
        # EXERCISE_JOINTS[exercise]의 "첫 번째 관절"을 기준으로 삼기 때문
        # (팔굽혀펴기는 팔꿈치 각도가 반복 동작의 저점/정점을 가장 뚜렷하게 나타냄).
        "elbow": {
            "left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        },
        # 허리(엉덩이가 처지거나 솟는 정도): 어깨-엉덩이-발목이 일직선(180도에 가까움)이어야
        # 정상 폼. 엉덩이가 아래로 처지든 위로 솟든 이 각도가 180도에서 멀어진다
        # (다만 부호가 없는 각도라 "처짐"과 "솟음"을 방향까지 구분하지는 못하고,
        # "몸통이 일직선에서 벗어났다"는 정도만 잡는다).
        "hip": {
            "left": (LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),
            "right": (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE),
        },
        # 다리(무릎이 굽혀지는 정도): 엉덩이-무릎-발목. 무릎을 굽히거나 바닥에 대는 등
        # 다리가 일자로 안 펴지는 자세를 잡아낸다.
        "knee": {
            "left": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            "right": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        },
        # 어깨(팔꿈치가 몸통에서 벌어지는 정도): 팔꿈치-어깨-엉덩이. 팔꿈치가 과도하게
        # 벌어지거나(flare) 반대로 너무 몸에 붙는 자세를 잡아낸다.
        "shoulder": {
            "left": (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
            "right": (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
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


def select_reliable_side(
    sequence: np.ndarray, exercise: str, joint_name: str | None = None, visibility_channel: int = -1
) -> str:
    """
    측면 촬영에서는 카메라에 가까운 쪽 팔/다리가 먼 쪽을 가리거나, 먼 쪽이 몸통에
    가려져 visibility가 구조적으로 낮게 나오는 경우가 흔하다 (예: pushup 영상에서
    한쪽 팔꿈치가 영상 전체 내내 visibility<0.5인 케이스가 실제로 확인됨).

    시퀀스 전체(또는 rep 하나)에서 좌/우 각각을 구성하는 관절들의 평균 visibility를
    비교해 더 신뢰할 수 있는 쪽을 고른다.

    Args:
        sequence: (T, J, C) ndarray. visibility_channel=-1(기본값)은 항상 "마지막 채널"을
            의미하므로, (x,y,visibility) 3채널이든 (x,y,z,visibility) 4채널이든
            둘 다 그대로 잘 동작한다.
        exercise: EXERCISE_JOINTS에 정의된 운동
        joint_name: 비교할 관절 이름 (None이면 EXERCISE_JOINTS[exercise]의 첫 번째 관절 사용)

    Returns:
        "left" 또는 "right"
    """
    if exercise not in EXERCISE_JOINTS:
        raise ValueError(f"정의되지 않은 exercise: {exercise}")

    if joint_name is None:
        joint_name = next(iter(EXERCISE_JOINTS[exercise]))
    sides = EXERCISE_JOINTS[exercise][joint_name]

    mean_visibility = {
        side: float(np.mean(sequence[:, [a, b, c], visibility_channel]))
        for side, (a, b, c) in sides.items()
    }
    return max(mean_visibility, key=mean_visibility.get)