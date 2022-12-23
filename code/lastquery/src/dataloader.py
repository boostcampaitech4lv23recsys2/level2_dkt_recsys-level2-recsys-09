import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ['assessmentItemID', 'testId','KnowledgeTag','cate2' , 'month'] 
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)
        df["Timestamp"] = df["Timestamp"].astype(str)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self,train_df):
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        train_df.sort_values(by=['userID','Timestamp'], inplace=True)
        train_df['cate2']=train_df.assessmentItemID.apply(lambda x:x[2])
    
    
        train_df['Timestamp']=pd.to_datetime(train_df['Timestamp'])
        train_df['month'] = train_df['Timestamp'].dt.month
        train_df['hour'] = train_df['Timestamp'].dt.hour
        
        
        diff = train_df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
        diff = diff.fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

        train_df['elapsed'] = diff  
        
        train_df['elapsed']=np.where(train_df['elapsed']>650,np.nan,train_df['elapsed'])
        
        q3=train_df['userID'].map(train_df.groupby('userID').quantile(0.75)['elapsed'])
        train_df['elapsed']=np.where(train_df['elapsed'].isna(),q3,train_df['elapsed'])
        
        accuracy_by_assess = train_df.groupby('assessmentItemID')['answerCode'].mean()
        accuracy_by_test = train_df.groupby('testId')['answerCode'].mean()
        accuracy_by_tags = train_df.groupby('KnowledgeTag')['answerCode'].mean()
        
        train_df['accuracy_assess'] = train_df['assessmentItemID'].map(accuracy_by_assess)
        train_df['accuracy_test'] = train_df['testId'].map(accuracy_by_test)
        train_df['accuracy_tags'] = train_df['KnowledgeTag'].map(accuracy_by_tags)
        
        train_df['relative_answered_correctly_assess'] = train_df['answerCode'] - train_df['accuracy_assess']
        train_df['relative_answered_correctly_tag'] = train_df['answerCode'] - train_df['accuracy_test']
        train_df['relative_answered_correctly_test'] = train_df['answerCode'] - train_df['accuracy_tags']
        userbymean=train_df.groupby('userID').mean()
        userid2rela=userbymean['relative_answered_correctly_assess']+userbymean['relative_answered_correctly_tag']+userbymean['relative_answered_correctly_test']
        
        train_df['total_rela']=train_df['userID'].map(userid2rela)
        
        tagdata=train_df.copy()
        tagdata['cumtag']=1
        tagdata=tagdata.groupby(['userID','KnowledgeTag']).cumsum()
        tagdata=tagdata.shift(1)
        idx=train_df['userID'].diff()!=0
        tagdata[idx]=0
        train_df['cumtag_ans']=tagdata['cumtag']
        train_df['cumtag']=tagdata['answerCode']
        train_df['cumtag_ratio']=train_df['cumtag']/train_df['cumtag_ans']
        
        train_df['Timestamp']=pd.to_datetime(train_df['Timestamp'])
        diff = train_df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
        diff = diff.fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

        train_df['elapsed'] = diff  

        train_df['elapsed']=np.where(train_df['elapsed']>650,np.nan,train_df['elapsed'])

        q3=train_df['userID'].map(train_df.groupby('userID').quantile(0.75)['elapsed'])
        train_df['elapsed']=np.where(train_df['elapsed'].isna(),q3,train_df['elapsed']) 
        
        train_df.fillna(0, inplace=True)
        
        return train_df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_test = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tag = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )
        self.args.n_cate2 = len(
            np.load(os.path.join(self.args.asset_dir, "cate2_classes.npy"))
        )
        self.args.n_month = len(
            np.load(os.path.join(self.args.asset_dir, "month_classes.npy"))
        )
        

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", 'total_rela','cumtag','cumtag_ratio','elapsed',
                   'accuracy_assess','accuracy_test', 'accuracy_tags',
                   'assessmentItemID', 'testId','KnowledgeTag','cate2' , 
                   'month', 'answerCode' ]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
           
                lambda r: (
                    r['total_rela'].values, 
                    r['cumtag'].values,  
                    r['cumtag_ratio'].values,  
                    r['elapsed'].values, 
                    r['accuracy_assess'].values, 
                    r[ 'accuracy_test'].values, 
                    r[ 'accuracy_tags'].values,
                    r['assessmentItemID'].values, 
                    r[ 'testId'].values, 
                    r[ 'KnowledgeTag'].values, 
                    r[ 'cate2' ].values,  
                    r[ 'month'].values,  
                    r['answerCode'].values, 
                )
            )
        )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        a1,a2,a3,a4,a5,a6,a7,c1,c2,c3,c4,c5,correct = row

        cate_cols = [a1,a2,a3,a4,a5,a6,a7,c1,c2,c3,c4,c5,correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader



def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = args.stride

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1
            
            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(col[window_i*stride:window_i*stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))


    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas


def data_augmentation(data, args):
    if args.window == True:
        data = slidding_window(data, args)

    return data
        