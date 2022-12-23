import os
from config import CFG
import pandas as pd
import torch


def prepare_dataset(device, basepath, verbose=True, logger=None):
    data = load_data(basepath)
    train_data, valid_data, test_data = separate_data(data)
    id2index = indexing_data(data)
    train_data_proc = process_data(train_data, id2index, device)
    valid_data_proc = process_data(valid_data, id2index, device)
    test_data_proc = process_data(test_data, id2index, device)
    

    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)

    return train_data_proc, valid_data_proc,test_data_proc, len(id2index)


def load_data(basepath):
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    data = pd.concat([data1, data2])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )

    return data


def separate_data(data):
    # train_data = data[data.answerCode >= 0]
    # test_data = data[data.answerCode < 0]
    
    val_idx=(data[data.answerCode < 0].index-1).tolist()
    train_data=data[data.answerCode >= 0].drop(val_idx, axis=0)
    valid_data=data.iloc[val_idx,:]
    test_data=data[data.answerCode < 0]

    return train_data, valid_data, test_data


def indexing_data(data):
    userid, itemid, testid, tagid= (
    sorted(list(set(data.userID))),
    sorted(list(set(data.assessmentItemID))),
    sorted(list(set(data.testId))),
    sorted(list(set(data.KnowledgeTag))),
)
    n_user, n_item ,n_test, n_tag= len(userid), len(itemid), len(testid), len(tagid)
    CFG.n_user=n_user
    CFG.n_item=n_item
    CFG.n_test=n_test
    CFG.n_tag=n_tag
    
    itemid_2_index = {v: i for i, v in enumerate(userid)}
    userid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    
    id_2_index = dict(itemid_2_index, **userid_2_index)

    testid_2_idx = {v: i for i, v in enumerate(testid)}
    tagid_2_idx = {v: i for i, v in enumerate(tagid)}

    tmp=data.drop_duplicates(
        subset=['assessmentItemID', 'testId', 'KnowledgeTag'], keep="last")
    itemid_info=tmp['assessmentItemID'].map(itemid_2_index).values
    testid_info=tmp['testId'].map(testid_2_idx).values
    tagid_info=tmp['KnowledgeTag'].map(tagid_2_idx).values

    CFG.itemid_info=itemid_info
    CFG.testid_info=testid_info
    CFG.tagid_info=tagid_info
    
    return id_2_index


def process_data(data, id_2_index, device):
    edge, label = [], []
    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id_2_index[user], id_2_index[item]
        # user와 item 사이의 edge
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)

    return dict(edge=edge.to(device), label=label.to(device))



def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
