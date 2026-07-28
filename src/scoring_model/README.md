# scoring_model

전처리된 특징/시퀀스를 입력받아 동작 점수를 산출하는 모델.

- baseline: 규칙 기반 + MLP
- labeling 된 데이터 구하기 힘듬 -> 규칙 기반 채점 + (DTW 거리 기반 or
-                                                   Z-Score or
-                                                   오토인코더 재구성 오차 or
-                                                  One-class SVM)

