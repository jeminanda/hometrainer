import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter # 애니메이션 관련 추가
from typing import List, Tuple

# MediaPipe Pose 주요 연결 선 (뼈대 연결 정보) - 기존과 동일
POSE_CONNECTIONS = [
    # 상체 (어깨, 팔, 골반)
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    # 하체 (골반, 무릎, 발목)
    (23, 25), (25, 27), (24, 26), (26, 28)
]

def animate_skeleton_3d(landmarks_sequence: np.ndarray, save_path: str = None, title: str = "3D Skeleton Animation"):
    """
    [시각화 3] 정규화된 3D 랜드마크 시퀀스(N_frames, 33, 3)를 애니메이션으로 시각화합니다.
    Jupyter Notebook에서 보거나 GIF/MP4로 저장할 수 있습니다.
    
    landmarks_sequence: shape (N_frames, 33, 3) [X, Y, Z]
    save_path: 저장할 경로 (예: 'animation.gif' 또는 'animation.mp4'). None이면 저장 안 함.
    """
    N_frames = landmarks_sequence.shape[0]
    
    # 1. Figure 및 3D Axes 초기화
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 상하 반전 처리 (MediaPipe Y축 방향 보정)
    ax.invert_yaxis()
    
    # 좌표축 범위 설정 (정규화된 데이터에 맞게 적절히 조절 필요, 예: -1.5 ~ 1.5)
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_ylim3d([-1.0, 1.0])
    ax.set_zlim3d([-1.0, 1.0])
    
    # 2. 초기 플롯 객체 생성 (빈 선 객체들)
    # 각 뼈대 연결선마다 하나의 Line3D 객체 생성
    lines = []
    for _ in POSE_CONNECTIONS:
        line, = ax.plot([], [], [], color='blue', linewidth=2)
        lines.append(line)
        
    # 원점(0,0,0) 골반 중심점 강조
    ax.scatter([0], [0], [0], c='green', marker='^', s=80, label='Origin (Mid-Hip)')
    ax.legend(loc='upper right')

    # 3. 매 프레임마다 호출될 업데이트 함수 정의
    def update(frame_idx):
        frame_landmarks = landmarks_sequence[frame_idx]
        
        xs = frame_landmarks[:, 0]
        ys = frame_landmarks[:, 1]
        zs = frame_landmarks[:, 2]
        
        # 각 선 객체의 XYZ 데이터 업데이트
        for i, (start_idx, end_idx) in enumerate(POSE_CONNECTIONS):
            # i번째 Line3D 객체 선택
            line = lines[i]
            
            # 시작점과 끝점의 XYZ 좌표를 Line3D 객체에 설정
            line.set_data([xs[start_idx], xs[end_idx]], [ys[start_idx], ys[end_idx]])
            line.set_3d_properties([zs[start_idx], zs[end_idx]])
            
        return lines # 업데이트된 플롯 객체 반환

    # 4. FuncAnimation 생성
    # interval: 프레임 간 지연 시간 (ms), 25fps ~ 40ms, 30fps ~ 33ms
    ani = FuncAnimation(fig, update, frames=N_frames, interval=33, blit=True)
    
    # 5. 애니메이션 저장 또는 표시
    if save_path:
        print(f"애니메이션을 저장 중입니다: {save_path} ...")
        # GIF로 저장하려면 PillowWriter, MP4로 저장하려면 FFMpegWriter 등 사용
        if save_path.endswith('.gif'):
            writer = PillowWriter(fps=1)
            ani.save(save_path, writer=writer)
        elif save_path.endswith('.mp4'):
            # mp4 저장엔 ffmpeg가 설치되어 있어야 함
            ani.save(save_path, fps=1, extra_args=['-vcodec', 'libx264'])
        print("저장이 완료되었습니다.")
        
    # Jupyter Notebook에서 애니메이션을 표시하려면 'jshtml'로 반환 (별도 설치 필요할 수 있음)
    # 혹은 그냥 plt.show()를 호출 (팝업 창으로 뜸)
    # plt.show() 
    
    return ani # Notebook에서 표시하기 위해 반환