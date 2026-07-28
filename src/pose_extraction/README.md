# pose_extraction

영상/이미지에서 MoveNet 또는 BlazePose를 이용해 관절 키포인트 좌표를 추출하는 모듈.

- 입력: 비디오 프레임 (이미지)
- 출력: 키포인트 좌표 시퀀스 (T, J, 2) + confidence
