from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# -------------------------------------------------------------
# MediaPipe BlazePose (33 Keypoints) 상세 뼈대 연결 정보
# -------------------------------------------------------------

FACE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),  # 오른쪽 얼굴/귀
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),  # 왼쪽 얼굴/귀
    (9, 10),  # 입술
]

UPPER_BODY_CONNECTIONS = [
    (11, 12),  # 어깨 연결
    (11, 13),
    (13, 15),  # 오른쪽 팔
    (12, 14),
    (14, 16),  # 왼쪽 팔
]

HAND_CONNECTIONS = [
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),  # 오른손
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),  # 왼손
]

TORSO_CONNECTIONS = [
    (11, 23),
    (12, 24),  # 옆구리
    (23, 24),  # 골반 라인
]

LOWER_BODY_CONNECTIONS = [
    (23, 25),
    (25, 27),  # 오른다리
    (24, 26),
    (26, 28),  # 왼다리
]

FOOT_CONNECTIONS = [
    (27, 29),
    (27, 31),
    (29, 31),  # 오른발
    (28, 30),
    (28, 32),
    (30, 32),  # 왼발
]

# 신체 부위별 색상 및 라인 두께
CONNECTION_GROUPS = [
    (FACE_CONNECTIONS, "gray", 1.5, "Face"),
    (UPPER_BODY_CONNECTIONS, "red", 2.5, "Arms/Shoulders"),
    (HAND_CONNECTIONS, "orange", 1.5, "Hands"),
    (TORSO_CONNECTIONS, "black", 3.0, "Torso"),
    (LOWER_BODY_CONNECTIONS, "blue", 2.5, "Legs"),
    (FOOT_CONNECTIONS, "green", 2.0, "Feet"),
]


def animate_skeleton_2d(
    landmarks_sequence: np.ndarray,
    save_path: str = "tests/",
    title: str = "2D Skeleton Animation",
):
    """[시각화] 2D 랜드마크 시퀀스(N_frames, 33, 2 이상의 차원)를 2D 평면 상에서 애니메이션으로 시각화합니다.

    landmarks_sequence: shape (N_frames, 33, C) -> 최소 [X, Y] 포함 save_path:
    저장할 경로 (.gif 또는 .mp4)
    """
    N_frames = landmarks_sequence.shape[0]

    # 1. Figure 및 2D Axes 초기화
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.axis('equal')

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.5)

    # MediaPipe Y축 보정 (이미지 좌표계 특성상 Y축이 아래로 증가하므로 반전)
    ax.invert_yaxis()

    # X, Y 축 범위 자동 설정 (여백 포함)
    x_min, x_max = (
        np.nanmin(landmarks_sequence[:, :, 0]),
        np.nanmax(landmarks_sequence[:, :, 0]),
    )
    y_min, y_max = (
        np.nanmin(landmarks_sequence[:, :, 1]),
        np.nanmax(landmarks_sequence[:, :, 1]),
    )

    x_margin = (x_max - x_min) * 0.15 if (x_max - x_min) > 0 else 0.5
    y_margin = (y_max - y_min) * 0.15 if (y_max - y_min) > 0 else 0.5

    ax.set_xlim([x_min - x_margin, x_max + x_margin])
    ax.set_ylim([y_max + y_margin, y_min - y_margin])  # Y축 반전 반영

    # 2. 그룹별 Line2D 및 Scatter 객체 생성
    lines_dict = {}
    for group_idx, (connections, color, lw, label) in enumerate(
        CONNECTION_GROUPS
    ):
        group_lines = []
        for _ in connections:
            (line,) = ax.plot([], [], color=color, linewidth=lw)
            group_lines.append(line)
        lines_dict[group_idx] = group_lines

        # 범례용 더미 그래프
        ax.plot([], [], color=color, linewidth=lw, label=label)

    # 관절 점 플롯
    (joints_scatter,) = ax.plot(
        [], [], "o", color="purple", markersize=4, label="Joints"
    )

    # 원점(0,0) 표기 (정규화 기준점)
    ax.scatter(
        [0],
        [0],
        c="gold",
        marker="^",
        s=100,
        edgecolors="black",
        label="Origin (0,0)",
        zorder=5,
    )
    ax.legend(loc="upper right", fontsize=9)

    # 3. 프레임 업데이트 함수
    def update(frame_idx):
        frame_landmarks = landmarks_sequence[frame_idx]

        xs = frame_landmarks[:, 0]
        ys = frame_landmarks[:, 1]

        updated_elements = []

        # 관절 점 업데이트
        joints_scatter.set_data(xs, ys)
        updated_elements.append(joints_scatter)

        # 뼈대 선 업데이트
        for group_idx, (connections, _, _, _) in enumerate(CONNECTION_GROUPS):
            group_lines = lines_dict[group_idx]
            for i, (start_idx, end_idx) in enumerate(connections):
                line = group_lines[i]
                line.set_data(
                    [xs[start_idx], xs[end_idx]], [ys[start_idx], ys[end_idx]]
                )
                updated_elements.append(line)

        return updated_elements

    # 4. FuncAnimation 생성
    ani = FuncAnimation(
        fig, update, frames=N_frames, interval=33, blit=True
    )

    # 5. 애니메이션 저장
    if save_path:
        print(f"2D 애니메이션 저장 중: {save_path} ...")
        if save_path.endswith(".gif"):
            writer = PillowWriter(fps=15)
            ani.save(save_path, writer=writer)
        elif save_path.endswith(".mp4"):
            ani.save(save_path, fps=30, extra_args=["-vcodec", "libx264"])
        print("저장 완료!")

    return ani