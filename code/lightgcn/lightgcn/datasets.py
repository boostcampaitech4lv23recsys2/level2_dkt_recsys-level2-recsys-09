import os

import pandas as pd
import torch

# 최종 데이터를 반환하는 함수 
def prepare_dataset(device, basepath, verbose=True, logger=None):
    data = load_data(basepath)
    train_data, test_data = separate_data(data)
    id2index = indexing_data(data)
    train_data_proc = process_data(train_data, id2index, device)
    test_data_proc = process_data(test_data, id2index, device)

    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)

    # train과 test, 그리고 인덱싱한 길이 반환 ~ 노드 수
    return train_data_proc, test_data_proc, len(id2index)

# 데이터 불러오며 동시에 train과 test를 합쳐버리고 문제/유저 중복값은 제거
def load_data(basepath):
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    # 합치고 제거
    data = pd.concat([data1, data2])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True)
    # 추가 피쳐 
    data['sub_category1'] = data['assessmentItemID'].str[2]
    data['sub_category2'] = data['assessmentItemID'].str[4:7]
    
    # data 반환
    return data

# train test 나누기
def separate_data(data):
    # 기준은 당연히 ans에 따라 진행 
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]

    # train, test 반환
    return train_data, test_data

# user와 item에 대해서 인덱싱 ~ 정형적으로 되어있지는 않아서 
def indexing_data(data):
    userid, itemid, sub1id, sub2id = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
        sorted(list(set(data.sub_category1))),
        sorted(list(set(data.sub_category2)))
    )
    n_user, n_item, n_sub1, n_sub2 = len(userid), len(itemid), len(sub1id), len(sub2id)

    # 인덱싱 번호를 점점 늘리는 것은 각 feature별로 고유한 인덱스가 아니라 전체 개별 인덱스를 위해서
    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    sub1_2_index = {v: i + n_user + n_item for i, v in enumerate(sub1id)}
    sub2_2_index = {v: i + n_user + n_item + n_sub1 for i, v in enumerate(sub2id)}
    id_2_index = {'user':userid_2_index, 'item':itemid_2_index,'sub1':sub1_2_index, 'sub2':sub2_2_index}
    # 인덱스를 반환
    return id_2_index

# 데이터를 노드로 만드는 과정 
def process_data(data, id_2_index, device):
    # 각각 노드와 노드에 대한 라벨(정답여부)
    edge, label = [], []
    for user, item, sub1, sub2 ,acode in zip(data.userID, data.assessmentItemID, data.sub_category1, data.sub_category2 ,data.answerCode):
        # 각각 모든 데이터에 대해서 인덱싱을 하고 노드와 라벨에 저장
        uid, iid, sid1, sid2 = id_2_index['user'][user], id_2_index['item'][item], id_2_index['sub1'][sub1], id_2_index['sub2'][sub2]
        edge.append([uid, [iid, sid1, sid2]])
        label.append(acode)

    # 노드와 라벨에 대한 리스트를 텐서로
    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)

    # dict으로 반환
    return dict(edge=edge.to(device), label=label.to(device))

# verbose를 위한 데이터 크기 확인
def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
