"""
src/pose_extraction/blazepose_extractor.py

BlazePose(mediapipe Tasks API, mediapipe>=1.0.0)를 이용해 영상에서
프레임별 관절 키포인트 좌표를 추출하는 모듈.

mediapipe 1.0.0부터는 legacy `mp.solutions.pose` 대신 안정화된
Tasks API(`mediapipe.tasks.python.vision.PoseLandmarker`)를 사용한다.
PoseLandmarker는 .task 모델 번들(pose_landmarker_lite/full/heavy.task)을
로드해서 동작하며, RunningMode.VIDEO 모드로 프레임 타임스탬프 기반 추론을 한다.

파이프라인 상 위치:
    영상 프레임 -> [pose_extraction] (본 모듈) -> 키포인트 시퀀스 (T, J, C)
                -> [preprocessing] -> [scoring_model]

출력 포맷:
    - keypoints: np.ndarray, shape (T, J, C)
        T: 프레임 수
        J: 관절(랜드마크) 개수 (BlazePose 기본 33개)
        C: 좌표 차원 (기본 x, y, visibility = 3. z 포함 시 4)
    - 프레임별로 포즈 미검출 시 NaN으로 채움 (후처리 단계에서 보간 처리 가정)

필요 패키지:
    pip install mediapipe>=1.0.0 opencv-python numpy

모델 다운로드 (예: full 모델):
    wget -O pose_landmarker_full.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )
except ImportError:  # pragma: no cover
    mp = None
    BaseOptions = None
    PoseLandmarker = None
    PoseLandmarkerOptions = None
    RunningMode = None


# -----------------------------
# 설정 / 데이터 구조
# -----------------------------

@dataclasses.dataclass
class ExtractionConfig:
    """BlazePose(PoseLandmarker) 추출 관련 설정."""

    model_path: str = "models/pose_landmarker_full.task"  # .task 모델 번들 경로
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    output_segmentation_masks: bool = False
    target_fps: Optional[int] = None       # None이면 원본 fps 그대로 사용, 지정 시 프레임 샘플링
    include_z: bool = False                # True면 (x, y, z, visibility), False면 (x, y, visibility)
    min_detected_ratio: float = 0.1        # 이 비율보다 검출률이 낮으면 PoseNotDetectedError
    use_gpu: bool = False                  # True면 GPU delegate 사용 (지원 안 되는 환경에선 CPU로 자동 대체)


@dataclasses.dataclass
class FrameKeypoints:
    """단일 프레임의 추출 결과."""

    frame_idx: int
    timestamp_ms: int
    landmarks: Optional[np.ndarray]  # shape (J, C), 미검출 시 None
    detected: bool


class PoseNotDetectedError(RuntimeError):
    """영상 전체(또는 대부분)에서 포즈가 검출되지 않았을 때 발생."""


# -----------------------------
# 메인 추출기
# -----------------------------

class BlazePoseExtractor:
    """
    영상 파일 또는 프레임 스트림에서 BlazePose(PoseLandmarker) 키포인트를
    추출하는 클래스. mediapipe 1.0.0 Tasks API 기준.

    사용 예시:
        with BlazePoseExtractor(ExtractionConfig(model_path="models/pose_landmarker_full.task")) as extractor:
            keypoints, meta = extractor.extract_from_video("data/raw/pushup_001.mp4")
            np.save("data/raw/pushup_001_keypoints.npy", keypoints)
    """

    NUM_LANDMARKS = 33  # BlazePose 랜드마크 개수

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._landmarker: Optional["PoseLandmarker"] = None

    # ---- 내부: 모델 초기화 ----

    def _load_model(self, running_mode: "RunningMode"):
        """PoseLandmarker 인스턴스를 지정된 running_mode로 생성/재사용."""
        if mp is None:
            raise ImportError(
                "mediapipe가 설치되어 있지 않습니다. `pip install mediapipe>=1.0.0` 을 실행하세요."
            )

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"PoseLandmarker 모델 파일을 찾을 수 없습니다: {model_path}. "
                "https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models "
                "에서 .task 파일을 받아 config.model_path에 지정하세요."
            )

        delegate = BaseOptions.Delegate.GPU if self.config.use_gpu else BaseOptions.Delegate.CPU

        try:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path), delegate=delegate),
                running_mode=running_mode,
                num_poses=self.config.num_poses,
                min_pose_detection_confidence=self.config.min_pose_detection_confidence,
                min_pose_presence_confidence=self.config.min_pose_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                output_segmentation_masks=self.config.output_segmentation_masks,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
        except Exception as e:
            if self.config.use_gpu:
                # 일부 환경(특히 Windows)에서는 mediapipe Tasks API의 GPU delegate가
                # 아직 완전히 지원되지 않을 수 있다. GPU 초기화가 실패하면 조용히 넘어가지
                # 않고 CPU로 자동 폴백하되, 사용자가 알 수 있게 경고를 남긴다.
                print(f"[경고] GPU delegate 초기화 실패({e}) -> CPU로 대체합니다.")
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
                    running_mode=running_mode,
                    num_poses=self.config.num_poses,
                    min_pose_detection_confidence=self.config.min_pose_detection_confidence,
                    min_pose_presence_confidence=self.config.min_pose_presence_confidence,
                    min_tracking_confidence=self.config.min_tracking_confidence,
                    output_segmentation_masks=self.config.output_segmentation_masks,
                )
                self._landmarker = PoseLandmarker.create_from_options(options)
            else:
                raise

    # ---- 공개 API ----

    def extract_from_video(
        self, video_path: str | Path
    ) -> tuple[np.ndarray, dict]:
        """
        영상 파일 전체를 처리하여 (T, J, C) 키포인트 배열과 메타데이터를 반환.

        Args:
            video_path: 입력 영상 경로 (data/raw/ 하위 파일 등)

        Returns:
            keypoints: np.ndarray, shape (T, J, C)
            meta: dict — fps, num_frames, detected_ratio, source_path 등

        Raises:
            PoseNotDetectedError: 검출률이 config.min_detected_ratio 미만인 경우
        """
        if cv2 is None:
            raise ImportError("opencv-python이 설치되어 있지 않습니다. `pip install opencv-python`")

        video_path = Path(video_path)
        if self._landmarker is None:
            self._load_model(RunningMode.VIDEO)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_fps = self.config.target_fps or source_fps
        sample_interval = max(1, round(source_fps / target_fps))

        results: list[FrameKeypoints] = []
        raw_idx = 0
        kept_idx = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if raw_idx % sample_interval == 0:
                    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    if timestamp_ms <= 0:
                        # 일부 코덱/컨테이너는 POS_MSEC을 못 주므로 fps 기반으로 fallback
                        timestamp_ms = int(kept_idx * (1000.0 / target_fps))
                    frame_result = self._extract_single_frame(
                        frame_bgr, frame_idx=kept_idx, timestamp_ms=timestamp_ms
                    )
                    results.append(frame_result)
                    kept_idx += 1

                raw_idx += 1
        finally:
            cap.release()

        keypoints, meta = self._stack_results(results)
        meta.update(
            {
                "fps": target_fps,
                "source_fps": source_fps,
                "source_path": str(video_path),
            }
        )
        return keypoints, meta

    def extract_from_frames(
        self, frames: Iterator[np.ndarray], fps: float
    ) -> tuple[np.ndarray, dict]:
        """
        이미 디코딩된 BGR 프레임 이터러블로부터 키포인트 추출.
        (예: 실시간 스트림, opencv로 이미 읽어둔 프레임 리스트 등)

        Args:
            frames: 프레임 이터러블, 각 원소는 (H, W, 3) BGR ndarray
            fps: 프레임 시퀀스의 기준 fps (타임스탬프 계산용)

        Returns:
            extract_from_video와 동일한 포맷의 (keypoints, meta)
        """
        if self._landmarker is None:
            self._load_model(RunningMode.VIDEO)

        results: list[FrameKeypoints] = []
        for idx, frame_bgr in enumerate(frames):
            timestamp_ms = int(idx * (1000.0 / fps))
            results.append(
                self._extract_single_frame(frame_bgr, frame_idx=idx, timestamp_ms=timestamp_ms)
            )

        keypoints, meta = self._stack_results(results)
        meta.update({"fps": fps, "source_fps": fps, "source_path": None})
        return keypoints, meta

    # ---- 내부: 단일 프레임 처리 ----

    def _extract_single_frame(
        self, frame_bgr: np.ndarray, frame_idx: int, timestamp_ms: int
    ) -> FrameKeypoints:
        """단일 프레임(BGR ndarray)에서 PoseLandmarker로 랜드마크를 추출."""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # RunningMode.VIDEO는 timestamp가 반드시 단조 증가해야 함
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return FrameKeypoints(
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                landmarks=None,
                detected=False,
            )

        # num_poses=1 기준 첫 번째 포즈만 사용
        landmarks = result.pose_landmarks[0]
        coords = self._landmarks_to_array(landmarks)

        return FrameKeypoints(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            landmarks=coords,
            detected=True,
        )

    def _landmarks_to_array(self, landmarks) -> np.ndarray:
        """PoseLandmarker의 NormalizedLandmark 리스트를 (J, C) ndarray로 변환."""
        if self.config.include_z:
            coords = [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]
        else:
            coords = [(lm.x, lm.y, lm.visibility) for lm in landmarks]
        return np.asarray(coords, dtype=np.float32)

    # ---- 내부: 후처리 ----

    def _stack_results(
        self, results: list[FrameKeypoints]
    ) -> tuple[np.ndarray, dict]:
        """FrameKeypoints 리스트를 (T, J, C) ndarray로 변환. 미검출 프레임은 NaN."""
        num_channels = 4 if self.config.include_z else 3
        empty_frame = np.full((self.NUM_LANDMARKS, num_channels), np.nan, dtype=np.float32)

        stacked = np.stack(
            [r.landmarks if r.detected else empty_frame for r in results],
            axis=0,
        ) if results else np.empty((0, self.NUM_LANDMARKS, num_channels), dtype=np.float32)

        num_frames = len(results)
        num_detected = sum(1 for r in results if r.detected)
        detected_ratio = (num_detected / num_frames) if num_frames > 0 else 0.0

        if num_frames == 0 or detected_ratio < self.config.min_detected_ratio:
            raise PoseNotDetectedError(
                f"검출률이 너무 낮습니다 (detected_ratio={detected_ratio:.2%}, "
                f"threshold={self.config.min_detected_ratio:.2%}). 영상/조명/프레이밍을 확인하세요."
            )

        meta = {
            "num_frames": num_frames,
            "num_detected": num_detected,
            "detected_ratio": detected_ratio,
        }
        return stacked, meta

    def close(self):
        """모델 리소스 해제."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# -----------------------------
# CLI 진입점 (단독 실행 시 data/raw -> npy 저장)
# -----------------------------

def main():
    """
    사용 예:
        python -m src.pose_extraction.blazepose_extractor \
            --input data/raw/pushup_001.mp4 \
            --output data/raw/pushup_001_keypoints.npy \
            --model models/pose_landmarker_full.task
    """
    import argparse

    parser = argparse.ArgumentParser(description="BlazePose(PoseLandmarker) 키포인트 추출")
    parser.add_argument("--input", required=True, help="입력 영상 경로")
    parser.add_argument("--output", required=True, help="출력 .npy 경로")
    parser.add_argument("--model", default="models/pose_landmarker_full.task", help=".task 모델 경로")
    parser.add_argument("--target-fps", type=int, default=None, help="샘플링 fps (기본: 원본 fps)")
    parser.add_argument("--include-z", action="store_true", help="z좌표 포함 여부")
    args = parser.parse_args()

    config = ExtractionConfig(
        model_path=args.model,
        target_fps=args.target_fps,
        include_z=args.include_z,
    )

    with BlazePoseExtractor(config) as extractor:
        keypoints, meta = extractor.extract_from_video(args.input)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, keypoints)
    print(f"저장 완료: {output_path} shape={keypoints.shape} meta={meta}")


if __name__ == "__main__":
    main()
