import gc
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class DataToSequence(Dataset):
    def __init__(self, groups, seq_len):
        # 개별 잘라진 시퀀스 저장 dict
        self.samples = {}
        self.seq_len = seq_len
        # 유저확인용 리스트
        self.user_ids = []

        # 모든 각 유저들에 대해
        for user_id in groups.index:
            item_id, test_id, time_lag, elaps_time,avg_user,avg_item,avg_tag,answer_code=groups[user_id]
            
            # 시퀀스의 길이가 1인 경우, 즉 문제풀이 row가 하나인 경우는 무시
            if len(item_id) < 2:
                continue
            
            # 주어진 시퀀스 길이보다 전체 유저의 시퀀스가 길 경우 자르기
            if len(item_id) > self.seq_len:
                initial = len(item_id) % self.seq_len
                if initial > 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (
                        item_id[:initial],test_id[:initial],time_lag[:initial],
                        elaps_time[:initial],avg_user[:initial],avg_item[:initial], 
                        avg_tag[:initial],answer_code[:initial]
                    )
                # 시퀀스만큼 몇번 진행되어야 하는지
                chunks = len(item_id)//self.seq_len
                # 남은 것들에 대해 다시 반복 진행
                for c in range(chunks):
                    start = initial + c*self.seq_len
                    end = initial + (c+1)*self.seq_len
                    self.user_ids.append(f"{user_id}_{c+1}")
                    self.samples[f"{user_id}_{c+1}"] = (
                        item_id[start:end],test_id[start:end],time_lag[start:end],
                        elaps_time[start:end], avg_user[start:end],avg_item[start:end],
                        avg_tag[start:end],answer_code[start:end]
                    )
            # 주어진 시퀀스 이하면 그냥 그대로 저장
            else:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (item_id,test_id,time_lag,elaps_time,avg_user,avg_item,avg_tag,answer_code)

    def __len__(self):
        return len(self.user_ids)

    # 하나씩 반환되는 부분
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        # 시퀀스별로 잘라진 애들에 대해
        item_id, test_id, time_lag, elaps_time, avg_user,avg_item,avg_tag,answer_code = self.samples[user_id]
        seq_len = len(item_id)

        # padding용 코드
        itemid = np.zeros(self.seq_len, dtype=int)
        testid = np.zeros(self.seq_len, dtype=int)
        timelag=np.zeros(self.seq_len, dtype=int)
        elapstime= np.zeros(self.seq_len, dtype=int)
        avguser = np.zeros(self.seq_len, dtype=int)
        avgitem = np.zeros(self.seq_len, dtype=int)
        avgtag = np.zeros(self.seq_len, dtype=int)
        answercode= np.zeros(self.seq_len, dtype=int)
        label = np.zeros(self.seq_len, dtype=int)

        if seq_len == self.seq_len:
            itemid[:] = item_id
            testid[:] = test_id
            timelag[:]=time_lag
            elapstime[:] = elaps_time
            avguser[:] = avg_user
            avgitem[:] = avg_item
            avgtag[:] = avg_tag
            answercode[:] = answer_code

        # padding
        else:
            itemid[-seq_len:] = item_id
            testid[-seq_len:] = test_id
            timelag[-seq_len:] = time_lag
            elapstime[-seq_len:] = elaps_time
            avguser[-seq_len:] = avg_user
            avgitem[-seq_len:] = avg_item
            avgtag[-seq_len:] = avg_tag
            answercode[-seq_len:] = answer_code

        # 직전 답으로 예상하는 시퀀스 to 시퀀스 예상을 위해서 답과 문제정보를 엇갈리게
        itemid=itemid[1:]
        testid=testid[1:]
        timelag=timelag[1:]
        elapstime=elapstime[1:]
        avguser=avguser[1:]
        avgitem=avgitem[1:]
        avgtag=avgtag[1:]
        # answer에 -1하고 clip하면 다 0되는데 forward backward 학습을 위해 랜덤으로 answer를 만드는 것과 동일한 것
        #FIXME fail label을 0-1 랜덤한 값으로 설정? 
        label = answercode[1:] 
        label = np.clip(label, 0, 1)
        answercode = answercode[:-1]
        
        #TODO testid를 반환하지 않고 쓰지도 않는다 ~ 나중에 확인 필요, 아마 knowledge tag도 나중에 확인 필요
        return itemid,timelag, elapstime, avguser,avgitem,avgtag,answercode, label
    
    
# submission을 위한 시퀀스 만들기
class DataToSequenceSub(Dataset):
    def __init__(self, groups, seq_len):
        # 개별 잘라진 시퀀스 저장 dict
        self.samples = {}
        self.seq_len = seq_len
        # 유저 확인용 리스트
        self.user_ids = []
        
        # 모든 각 유저들에 대해
        for user_id in groups.index:
            item_id, test_id, time_lag, elaps_time,avg_user,avg_item, avg_tag,answer_code=groups[user_id]
            
            # 시퀀스의 길이가 1인 경우, 즉 문제풀이 row가 하나인 경우는 무시
            if len(item_id) < 2:
                continue
            
            # 주어진 시퀀스 길이보다 전체 유저의 시퀀스가 길 경우 자르기
            # 이때 마지막 시퀀스에 대해서만 가져가면 되니 마지막 시퀀스만 시퀀스 길이로 잘라 가져가기
            if len(item_id) > self.seq_len:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (
                    item_id[-seq_len:],test_id[-seq_len:],time_lag[-seq_len:],
                    elaps_time[-seq_len:],avg_user[-seq_len:],avg_item[-seq_len:] ,
                    avg_tag[-seq_len:],answer_code[-seq_len:])
            else:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (item_id,test_id,time_lag,elaps_time,avg_user,avg_item,avg_tag,answer_code)
    
    
    def __len__(self):
        return len(self.user_ids)
    
    # 하나씩 반환되는 부분
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        # 각 마지막 시퀀스들에 대해
        item_id, test_id, time_lag, elaps_time, avg_user,avg_item,avg_tag,answer_code = self.samples[user_id]
        seq_len = len(item_id)

        # padding용 코드
        itemid = np.zeros(self.seq_len, dtype=int)
        testid = np.zeros(self.seq_len, dtype=int)
        timelag=np.zeros(self.seq_len, dtype=int)
        elapstime= np.zeros(self.seq_len, dtype=int)
        avguser = np.zeros(self.seq_len, dtype=int) 
        avgitem = np.zeros(self.seq_len, dtype=int)
        avgtag = np.zeros(self.seq_len, dtype=int)
        answercode= np.zeros(self.seq_len, dtype=int)
        label = np.zeros(self.seq_len, dtype=int)

        if seq_len == self.seq_len:
            itemid[:] = item_id
            testid[:] = test_id
            timelag[:]=time_lag
            elapstime[:] = elaps_time
            avguser[:] = avg_user
            avgitem[:] = avg_item
            avgtag[:] = avg_tag
            answercode[:] = answer_code
          
        # padding  
        else:
            itemid[-seq_len:] = item_id
            testid[-seq_len:] = test_id
            timelag[-seq_len:] = time_lag
            elapstime[-seq_len:] = elaps_time
            avguser[-seq_len:] = avg_user
            avgitem[-seq_len:] = avg_item
            avgtag[-seq_len:] = avg_tag
            answercode[-seq_len:] = answer_code

        itemid=itemid[1:]
        testid=testid[1:]
        timelag=timelag[1:]
        elapstime=elapstime[1:]
        avguser=avguser[1:]
        avgitem=avgitem[1:]
        avgtag=avgtag[1:]
        label = answercode[1:] 
        label = np.clip(label, 0, 1)
        answercode = answercode[:-1]
        return itemid, timelag, elapstime, avguser,avgitem,avgtag,answercode, label