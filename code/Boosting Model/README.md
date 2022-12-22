# LGBM
- 목표 : 수치형 Feature가 많다고 판단하여 Timestamp, AnswerCode를 기반으로 한 파생 변수 생성을 통해 모델의 성능 향상 
- 난이도라는 관점에서 접근하여 범주형 Feature 생성에 중점을 둬서 상대적으로 피쳐 수와 하이퍼파라미터에 덜 민감한 Catboost를 사용하려 했으나, 변수가 적을 것이라 판단
- 독립 변수와 종속 변수와의 관계를 통해서 약 20개 정도의 파생 변수를 생성하였으나 성능의 변화가 진전이 크게 없었음(약 LB 0.75~0.76)
- 따라서, EDA를 통해 유의미한 관계를 나타내는 변수들은 우선 모델에 적용하고 모델의 성능을 체크하기로 판단
- 최종적으로 2배 이상(Feature 40개 후반)의 파생 변수 생성을 토대로 모델 성능 향상에 기여 (LB 0.825)


## EDA
- 독립 변수와 종속 변수와의 관계를 시각화


## Preprocessing
- EDA를 통해 여러 파생 변수를 생성함
- 수치형 변수인 Timestamp 활용하여 Elapsed(걸린 시간)을 활용하여 다양한 파생 변수 생성
- 종속 변수인 AnswerCode 활용하여 기준이 될 수 있는 Test, Assessment_ID, User 등과 같은 변수들을 기준으로 평균 및 합을 포함한 통계적 파생 변수 생성


## Split Version 1 - Train + Test Set
- Data Leakage 포함이 허용된 대회여서 두 데이터 셋을 합쳐서 데이터 양을 증가시켜 모델의 성능에 좋은 영향을 기대
- 모델의 성능 측면에서는 Train dataset에서 Split 한 것 보다 좋지 않았음
- 두 데이터 셋의 분포 차이가 클 것이라 판단


## Split Version 2 - Train set
- Train dataset만 활용하여 Train & Valid Split 진행
- Ver.1 보다 성능이 좋았음
- 따라서, CV(Cross-Validation)에 적용

## CV(Cross-Validation) - User Split
- Future Data Leakage Issue - 문제 풀이 Domain의 특성상 큰 영향은 없을 것이라 판단
- 10 Fold, 5 Fold 진행 
- 결과 : 10Fold > 5Fold

## Optuna
- CV진행 시 함께 진행
- 각 Fold 별마다 Optuna를 적용하여 최적의 파라미터 탐색
