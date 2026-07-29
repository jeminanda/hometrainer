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

def extract_exercise_angles(row_data: dict, exercise_type: str = "squat") -> dict:
    """
    1개 프레임의 3D 랜드마크 데이터를 받아 주요 운동 관절 각도를 추출합니다.
    """
    # 예시: MediaPipe 33개 랜드마크 인덱스 기준
    # 힙-무릎-발목 각도 (스쿼트 핵심 각도)
    # 어깨-힙-무릎 각도 (푸쉬업/스쿼트 상체 각도)
    
    # TODO: row_data에서 해당 랜드마크 3D 좌표 추출 코드로 연결
    angles = {}
    if exercise_type == "squat":
        # angles['left_knee'] = calculate_angle(hip, knee, ankle)
        pass
    elif exercise_type == "pushup":
        # angles['left_elbow'] = calculate_angle(shoulder, elbow, wrist)
        pass
        
    return angles