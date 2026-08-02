# pose_extraction

영상에서 BlazePose(mediapipe Tasks API, mediapipe>=1.0.0)로 프레임별 관절 키포인트
좌표를 추출하는 모듈.

## 파일

- `blazepose_extractor.py` — `BlazePoseExtractor` 클래스, `ExtractionConfig` 설정, CLI 진입점

## 출력 포맷

```
keypoints: np.ndarray, shape (T, J, C)
    T: 프레임 수
    J: 관절(랜드마크) 개수 (BlazePose 기본 33개)
    C: 좌표 채널
        include_z=False(기본): (x, y, visibility) — 3채널
        include_z=True        : (x, y, z, visibility) — 4채널
```

미검출 프레임은 NaN으로 채워지며, 이후 `preprocessing`이 보간 처리합니다.

**주의**: `visibility`는 항상 마지막 채널입니다. `n_spatial_dims`(preprocessing 쪽 파라미터)를
채널 수와 헷갈리지 마세요 — 3채널이든 4채널이든 좌표 채널 수는 기본 2(x, y)로 취급합니다.
z가 실제로 있고 이를 각도 계산에 쓰고 싶다면 명시적으로 `n_spatial_dims=3`을 지정해야 합니다
(자동 추론하지 않습니다 — visibility를 z로 오인하는 버그를 방지하기 위함).

## 사용 예

```python
from src.pose_extraction.blazepose_extractor import BlazePoseExtractor, ExtractionConfig

config = ExtractionConfig(
    model_path="models/pose_landmarker_full.task",
    include_z=False,
    target_fps=None,       # None이면 원본 fps 그대로 사용
    min_detected_ratio=0.1,
    use_gpu=True,           # GPU delegate 사용 (미지원 환경이면 자동으로 CPU 폴백)
)

with BlazePoseExtractor(config) as extractor:
    keypoints, meta = extractor.extract_from_video("data/raw/squat_001.mp4")

np.save("data/raw/squat_001_keypoints.npy", keypoints)
```

**영상 여러 개를 순회할 때는 파일마다 새 `BlazePoseExtractor`를 생성해야 합니다.**
`PoseLandmarker`(VIDEO 모드)는 프레임 타임스탬프가 반드시 단조 증가해야 하는데,
같은 extractor를 여러 영상에 재사용하면 다음 영상의 타임스탬프가 0부터 다시 시작되어
이 제약을 어기게 됩니다.

## CLI (키워드 기반 배치 추출)

`notebooks/Test.ipynb`에 `EXERCISE_KEYWORD`(예: "pushup")로 폴더 내 영상을 찾아
`{EXERCISE_KEYWORD}_{원본파일명}_keypoints.npy` 규칙으로 일괄 저장하는 셀이 있습니다.
원본 영상 파일명이 `build_dataset`이 기대하는 명명 규칙과 안 맞아도, 저장 시
`EXERCISE_KEYWORD`를 접두사로 강제하기 때문에 원본 영상을 리네이밍할 필요가 없습니다.

## GPU 관련 참고

mediapipe Tasks API의 GPU delegate가 Windows 데스크톱 환경에서는 아직 완전히
지원되지 않을 수 있습니다. `use_gpu=True`인데 초기화가 실패하면 예외 없이 CPU로
자동 폴백하며 `[경고]` 메시지를 출력합니다 — 이 메시지가 뜨면 실제로는 GPU가
안 쓰이고 있다는 뜻입니다.
