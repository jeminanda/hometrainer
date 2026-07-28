# Home Training Scorer

비전 AI를 활용해 홈 트레이닝 동작(팔굽혀펴기, 스쿼트 등)의 수행도를 자동으로 채점하는 프로젝트입니다.

## 프로젝트 개요

1. 영상에서 포즈 추정 모델(MoveNet / BlazePose)로 관절 키포인트 좌표를 추출
2. 반복 동작(rep) 단위로 슬라이싱 및 전처리 (속도/길이 편차 정규화)
3. 전처리된 시퀀스를 채점 모델에 입력하여 동작 품질 점수 산출

> 엣지 디바이스(모바일) 포팅은 후순위 목표로, 현재는 포즈 추출 → 전처리 → 채점 모델 개발에 집중합니다.

## 폴더 구조

```
home-training-scorer/
├── data/
│   ├── raw/            # 원본 영상 / 원본 키포인트 추출 결과
│   └── processed/      # 슬라이싱·정규화가 끝난 학습용 데이터
├── src/
│   ├── pose_extraction/ # MoveNet/BlazePose 등으로 키포인트 추출하는 코드
│   ├── preprocessing/   # rep 슬라이싱, 위상 정규화, 각도 계산 등 전처리 코드
│   └── scoring_model/   # 채점 모델 정의, 학습, 추론 코드
├── notebooks/           # 탐색적 분석, 실험용 노트북
├── configs/             # 실험 설정 파일 (yaml/json 등)
├── tests/               # 유닛 테스트
├── requirements.txt
└── README.md
```

## 파이프라인

```
영상 프레임
   │
   ▼
[pose_extraction] MoveNet/BlazePose (고정, 미학습)
   │  → 키포인트 좌표 시퀀스 (T, J, 2)
   ▼
[preprocessing] rep 슬라이싱 + 위상 정규화 + 각도/ROM 특징 추출
   │  → 고정 길이 특징 벡터/시퀀스
   ▼
[scoring_model] 채점 모델 (MLP → 추후 Transformer 확장)
   │
   ▼
동작 점수 / 등급 출력
```

## 로드맵

- [ ] 포즈 키포인트 추출 파이프라인 구축
- [ ] rep 슬라이싱 및 위상 기반 정규화 구현
- [ ] 베이스라인 채점 모델(규칙 기반 + MLP) 개발
- [ ] (확장) 키포인트 시퀀스 기반 Transformer 채점 모델
- [ ] (후순위) 엣지 디바이스(모바일) 경량화 및 포팅

## 개발 환경

```bash
pip install -r requirements.txt
```

## 라이선스

TBD
