# Deep Knowledge Tracing using BERT

### 주요 변경 사항

- train, valid and test set split: test 데이터 중 마지막 시퀀스만 test data로 사용하고 나머지 데이터는 valid data로 사용. train data는 모두 학습에 활용
- 유저 기준으로 시퀀스를 생성하면 시퀀스의 개수가 너무 적고, 시퀀스의 길이도 max_seq_len의 최적값인 15~20와 큰 차이를 보여 버려지는 데이터가 많아 유저ID-테스트ID를 함께 기준으로 사용하여 시퀀스를 생성하여 학습에 활용
- BertConfig를 활용하여 Bert 모델 내부 파라미터 조정
- timestamp를 활용하여 연속형/범주형 시간 변수 추가, LightGCN의 그래프 임베딩(유저-아이템 상호작용 학습에 )을 변수로 추가
