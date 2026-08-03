"""
src/preprocessing/overlay_visualizer.py

원본 영상(mp4) 위에, BlazePose가 뽑은 좌표를 실제 픽셀 위치에 그대로 그려서
오버레이 영상을 만든다. animate_skeleton_2d()는 정규화된 좌표만으로 그린
추상적인 막대인간이라 "이게 진짜 이 영상에서 나온 결과"라는 확신을 주기
약한데, 이 모듈은 화면에 원본 그대로 보여주면서 그 위에 관절점이 실시간으로
따라다니는 걸 보여줘서 훨씬 직관적으로 신뢰를 준다.

data/raw(학습용 영상만 모아두는 폴더)나 tests(시연/검증용 영상 폴더)나 상관없이,
영상 파일과 그에 대응하는 keypoints .npy 경로만 넘기면 어디서든 동작한다.

주의: 여기서 쓰는 keypoints는 BlazePoseExtractor가 뽑은 "원본"(정규화 이미지 좌표,
0~1 범위) 그대로여야 한다. coordiante_normalization.normalize_landmarks()를 거친
좌표(힙 중심 정렬 + 스케일 정규화)는 더 이상 원본 영상 픽셀 위치와 대응되지 않으므로
이 용도로 쓸 수 없다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .visualizer import CONNECTION_GROUPS

# matplotlib 색상 이름(visualizer.py 기준) -> OpenCV가 쓰는 BGR 튜플로 변환
_COLOR_NAME_TO_BGR = {
    "gray": (128, 128, 128),
    "red": (0, 0, 255),
    "orange": (0, 165, 255),
    "black": (0, 0, 0),
    "blue": (255, 0, 0),
    "green": (0, 200, 0),
}
_JOINT_COLOR_BGR = (255, 0, 200)  # purple 계열


def overlay_skeleton_on_video(
    video_path: str | Path,
    keypoints_path: str | Path,
    output_path: str | Path,
    visibility_channel: int = -1,
    visibility_threshold: float = 0.3,
    joint_radius: int = 4,
    line_thickness: int = 2,
) -> Path:
    """
    원본 영상(video_path) 위에 keypoints_path(.npy, BlazePoseExtractor 원본 출력)의
    좌표를 실제 픽셀 위치로 환산해 그려서 output_path에 저장한다.

    Args:
        video_path: 원본 영상 경로 (data/raw든 tests든 아무 위치나 가능)
        keypoints_path: 그 영상에서 뽑은 원본 keypoints .npy (정규화 이미지 좌표,
            정규화/DTW를 거치지 않은 것)
        output_path: 오버레이 결과를 저장할 mp4 경로
        visibility_channel: visibility가 들어있는 채널 (-1이면 마지막 채널,
            (x,y,visibility) 3채널이든 (x,y,z,visibility) 4채널이든 안전하게 동작)
        visibility_threshold: 이 값보다 낮은 관절은 흐릿하게/생략해서 그림
            (실제로 신뢰도가 낮았던 부분을 숨기지 않고 오히려 "여기는 불확실했다"를
            보여주는 게 시연에서는 더 정직하다고 판단해, 아예 안 그리기보다는
            흐리게 그리는 쪽을 기본으로 한다 - draw_low_visibility 참고)
        joint_radius, line_thickness: 그리기 스타일

    Returns:
        저장된 output_path (Path 객체)
    """
    if cv2 is None:
        raise ImportError("opencv-python이 필요합니다. `pip install opencv-python`")

    video_path = Path(video_path)
    keypoints_path = Path(keypoints_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keypoints = np.load(keypoints_path)  # (T, 33, C), 원본 정규화 이미지 좌표(0~1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    num_frames = min(total_video_frames, keypoints.shape[0])
    if total_video_frames != keypoints.shape[0]:
        print(
            f"[경고] 영상 프레임 수({total_video_frames})와 keypoints 프레임 수"
            f"({keypoints.shape[0]})가 다릅니다. 앞쪽 {num_frames}프레임만 오버레이합니다 "
            "(추출 시 target_fps로 샘플링했다면 원래 있을 수 있는 차이입니다)."
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    for frame_idx in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break

        lm = keypoints[frame_idx]  # (33, C)
        vis = lm[:, visibility_channel] if lm.shape[-1] > 2 else np.ones(lm.shape[0])
        px = lm[:, 0] * frame_width
        py = lm[:, 1] * frame_height

        def _point(idx: int):
            if np.isnan(px[idx]) or np.isnan(py[idx]):
                return None
            return (int(px[idx]), int(py[idx]))

        def _alpha_for(idx_a: int, idx_b: int) -> float:
            v = min(vis[idx_a], vis[idx_b]) if not (np.isnan(vis[idx_a]) or np.isnan(vis[idx_b])) else 0.0
            return float(np.clip(v, 0.15, 1.0)) if v < visibility_threshold else 1.0

        overlay = frame.copy()
        for connections, color_name, _lw, _label in CONNECTION_GROUPS:
            color = _COLOR_NAME_TO_BGR.get(color_name, (255, 255, 255))
            for a_idx, b_idx in connections:
                pa, pb = _point(a_idx), _point(b_idx)
                if pa is None or pb is None:
                    continue
                alpha = _alpha_for(a_idx, b_idx)
                line_color = tuple(int(c * alpha) for c in color)
                cv2.line(overlay, pa, pb, line_color, line_thickness, lineType=cv2.LINE_AA)

        for idx in range(lm.shape[0]):
            p = _point(idx)
            if p is None:
                continue
            v = vis[idx] if not np.isnan(vis[idx]) else 0.0
            alpha = float(np.clip(v, 0.15, 1.0)) if v < visibility_threshold else 1.0
            joint_color = tuple(int(c * alpha) for c in _JOINT_COLOR_BGR)
            cv2.circle(overlay, p, joint_radius, joint_color, -1, lineType=cv2.LINE_AA)

        writer.write(overlay)

    cap.release()
    writer.release()

    print(f"오버레이 영상 저장 완료: {output_path} ({num_frames}프레임)")
    return output_path
