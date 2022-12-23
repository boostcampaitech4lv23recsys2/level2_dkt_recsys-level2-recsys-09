## LastQuery

- custom transformer encoder, LSTM, DNN로 세가지 구성으로 이루어진 모델. custom transformer model은 원래의 transformer와 달리 마지막 시퀀스의 입력을 쿼리벡터로 사용하는 방식. Riiid 대회에서 긴 seq일수록 좋은 validation auc를 가졌기 때문에, 인풋 시퀀스 당 마지막 질문에 대한 정답 여부만 예측하는 학습관계(마지막 assessmentItemId[query]과 다른 assessmentItemId[key])는 충분하다고 판단.
- custom transformer encoder에서는 모델이 assessmentItemId사이의 관계를 학습시키는 역할을 하고, LSTM에서는 sequential특징을 학습해 최근 활동에 더 큰 가중치를 둘 수 있게 함.
- 처음 사용했던 FE는 유저별로 태그 당 정답 누적합, 태그 당 누적정답률, 전체 정답률, 최근 정답률, 푸는데 걸린시간을 사용했고, 제출한 결과와 리더보드 결과가 많이 달라 FE과정에서 과적합 문제가 생겼다고 판단.
- 1등의 솔루션 처럼 feature를 최대한 사용하지 않는 방식으로 진행. categorical 변수로 assessmentItemId, KnowledgeTag, testId 사용했고, continuous변수로는 Elapsedtime을 이상치 처리 후 maximum value로 나눔.
- 성능이 좋지 않아 seq_len를 늘리면서 학습시켰고, 길이를 늘리면서 성능이 떨어지는 것을 해결하기 위해 sliding window를 통해 데이터 증강 방식을 사용함.

## LightGCN

- 베이스 라인의 LightGCN은 모델을 임포트 해 userID의 길이와 assessmentItemId의 길이를 합친 개수를 노드 수로 사용해 임베딩 하는 방식이였고, assessmentItemId에 KnowledgeTag와 TestId정보를 합쳐서 임베딩하기 위해 수정한LightGCN모델을 사용
- userId, assessmentItemId, KnowledgeTag, TestId에 대한 임베딩 매트릭스를 따로 만들어 임베딩 시킨 벡터들을 concat 해 LGCN의 인풋으로 넣어줬지만 결과가 기존 베이스라인보다 성능이 좋지 않아 다른 정보를 합치는 과정에서 assesmentItemId에 대한 정보들이 사라져 생긴 문제라고 생각함.
- 위 문제를 해결하기 위해 임베딩하는 과정에서 assessmentItemId의 임베딩 매트릭스에 가중치를 더 부여하는 방식을 사용 (weightedembedding version)
