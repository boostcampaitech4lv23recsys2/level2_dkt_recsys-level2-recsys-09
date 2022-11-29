import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

def data_processing(train,test):
    print("데이터 불러오는 중...")
    # map하기
    df_=pd.concat([train, test]).reset_index(drop=True)
    asses2idx = {v:i for i,v in enumerate(df_['assessmentItemID'].unique())}
    test2idx = {v:i for i,v in enumerate(df_['testId'].unique())}

    train['assessmentItemID'] = train['assessmentItemID'].map(asses2idx)
    test['assessmentItemID'] = test['assessmentItemID'].map(asses2idx)

    train['testId'] = train['testId'].map(test2idx)
    test['testId'] = test['testId'].map(test2idx)\

    # train은 train, test는 val로 사용하기 ~ 어차피 대동소이하다    
    # train을 위해서 -1인 target row는 삭제
    # FIXME fail split방법 변경? 
    train_df = train[train['answerCode']>=0]
    val_df = test[test['answerCode']>=0]
    target_df = test[test['answerCode']<0]
    
    print("추가 피쳐 생성 완료!")
    
    # TODO 이때 knowledgetag는 사용하지 않는데 나중에 추가했을때 많이 달라지는 지 확인 필요
    features = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'Timestamp',
       'KnowledgeTag', 'time_lag', 'elapsed_time']
    print("현재 사용하는 FEATURE의 목록 : {}".format(features))

    print("사용을 위한 데이터 그룹핑...")
    train_group = train_df[features].groupby("userID").apply(lambda df: (
        df["assessmentItemID"].values,
        df["testId"].values,
        df['time_lag'].values,
        df["elapsed_time"].values,
        df["answerCode"].values
    ))

    val_group = val_df[features].groupby("userID").apply(lambda df: (
        df["assessmentItemID"].values,
        df["testId"].values,
        df['time_lag'].values,
        df["elapsed_time"].values,
        df["answerCode"].values
    ))
    print("데이터 처리 완료!")
    return train_group, val_group
    
def get_time_lag(df):
    # test Id별로 간격을 구하는 time lag
    time_dict = {}
    time_lag = np.zeros(len(df), dtype=np.float32)
    for idx, row in enumerate(df[["userID", "Timestamp", "testId"]].values):
        if row[0] not in time_dict:
            time_lag[idx] = 0
            time_dict[row[0]] = [row[1], row[2], 0] 
        else:
            if row[2] == time_dict[row[0]][1]:
                time_lag[idx] = time_dict[row[0]][2]
            else:
                time_lag[idx] = (datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S') - datetime.strptime(time_dict[row[0]][0], '%Y-%m-%d %H:%M:%S')).total_seconds()
                time_dict[row[0]][0] = row[1]
                time_dict[row[0]][1] = row[2]
                time_dict[row[0]][2] = time_lag[idx]

    df["time_lag"] = time_lag/60 
    df["time_lag"] = df["time_lag"].clip(0, 1440)
    return df

# FIXME 쓰지 말까?
def get_elapsed_time(df):
    # assin Id별로 간격을 구하는 elapsed time 
    elpased_dict = {}
    elapsed_time = np.zeros(len(df), dtype=np.float32)
    for idx, row in enumerate(df[["userID", "Timestamp", "assessmentItemID"]].values):
        if row[0] not in elpased_dict:
            elapsed_time[idx] = 0
            elpased_dict[row[0]] = [row[1], row[2], 0]
        else:
            if row[2] == elpased_dict[row[0]][1]:
                elapsed_time[idx] = elpased_dict[row[0]][2]
            else:
                elapsed_time[idx] = (datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S') - datetime.strptime(elpased_dict[row[0]][0], '%Y-%m-%d %H:%M:%S')).total_seconds()
                elpased_dict[row[0]][0] = row[1]
                elpased_dict[row[0]][1] = row[2]
                elpased_dict[row[0]][2] = elapsed_time[idx]

    df["elapsed_time"] = elapsed_time/60
    df["elapsed_time"] = df["elapsed_time"].clip(0, 1440)
    return df

# wandb용
def class2dict(f):
    return dict(
        (name, getattr(f, name)) for name in dir(f) if not name.startswith("__")
    )

