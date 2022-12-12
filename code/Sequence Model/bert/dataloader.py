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

    def split_data(self, data, ratio=0.8, shuffle=True, seed=42):
        """
        split data into two parts with a given ratio.
        *data의 경우 전처리된 상태: userID로 groupby되어 [testId, assessmentItemID, KnowledgeTag, answerCode]
        userid-testid 시퀀스를 유저별로 스플릿하려면, userid와 관련된 정보 필요
        """
        """
        # 모델 고도화를 위해 랜덤 셔플 방식 활용
        if shuffle:
            random.seed(seed)  # fix to default seed 42
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]
        """
        # 학습 데이터는 전부 학습에 활용
        data_1 = data
        # test data에서 마지막 시퀀스만 빼고 valid set으로 활용: 테스트 데이터에서 -1이 포함된 시퀀스 제외
        data_2 = []
        tmp = self.load_data_from_file_by_testid("test_data.csv", is_train=False)
        for i in range(len(tmp)):
            if -1 not in tmp[i][3]:
                data_2.append(tmp[i])

        print(f"Train Data: {len(data_1)}, Valid Data: {len(data_2)}")

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]    # "month", "day", "hour", "minute", "second"

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

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self, df):
        # TODO
        df['month'] = df["Timestamp"].str.replace('[^0-9]','', regex=True).map(lambda x: int(x[4:6]))
        df['day'] = df["Timestamp"].str.replace('[^0-9]','', regex=True).map(lambda x: int(x[6:8]))
        df['hour'] = df["Timestamp"].str.replace('[^0-9]','', regex=True).map(lambda x: int(x[8:10]))
        df['minute'] = df["Timestamp"].str.replace('[^0-9]','', regex=True).map(lambda x: int(x[10:12]))
        df['second'] = df["Timestamp"].str.replace('[^0-9]','', regex=True).map(lambda x: int(x[12:14]))

        return df

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

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                )
            )
        )

        return group.values

    def load_data_from_file_by_testid(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)
        """
        if is_train:    # 테스트 데이터를 학습에 활용
            test_df = pd.read_csv(os.path.join(self.args.data_dir, "test_data.csv"))
            df = pd.concat([df, test_df], ignore_index=True)
        """
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

        self.args.n_user = 7442 # user  (0~7441)

        self.args.n_month = 12  # month (1~12)
        self.args.n_day = 31    # day   (1~31)
        self.args.n_hour = 24   # hour  (0~23)
        self.args.n_minute = 60 # minute    (0~59)
        self.args.n_second = 60 # second    (0~59)

        df = df.sort_values(by=["userID", "testId", "Timestamp"], axis=0)
        #df["scaled_timestamp"] = (df["Timestamp"] - 1577804881) / (1609260381 - 1577804881) * 500  # max=1609260381, min=1577804881 / numerical
        columns = ["testId", "assessmentItemID", "KnowledgeTag", "answerCode", "userID", "month", "day", "hour", "minute", "second"]    # "scaled_timestamp"
        print(len(df), '=>', end=' ')
        group = (
            df[columns]
            .groupby(["userID", "testId"])
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                    r["userID"].values, # user

                    r["month"].values,  # month
                    r["day"].values,    # day
                    r["hour"].values,   # hour
                    r["minute"].values, # minute
                    r["second"].values, # second

                    #r["scaled_timestamp"].values, # scaled_timestamp
                )
            )
        )
        print(len(group.values))

        return group.values

    def load_train_data(self, file_name):
        """
        # -1이 포함된 시퀀스 제외
        self.train_data = []
        tmp = self.load_data_from_file_by_testid(file_name)
        for i in range(len(tmp)):
            if -1 not in tmp[i][3]:
                self.train_data.append(tmp[i])
        """
        self.train_data = self.load_data_from_file_by_testid(file_name)

    def load_test_data(self, file_name):
        # -1이 포함된 시퀀스만 반환
        self.test_data = []
        tmp = self.load_data_from_file_by_testid(file_name, is_train=False)
        for i in range(len(tmp)):
            if -1 in tmp[i][3]:
                self.test_data.append(tmp[i])


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct, user, month, day, hour, minute, second = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]   # stime / row[10]

        cate_cols = [test, question, tag, correct, user, month, day, hour, minute, second]  # stime

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
