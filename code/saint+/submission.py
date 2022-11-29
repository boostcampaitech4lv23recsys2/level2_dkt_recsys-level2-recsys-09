import time
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model import SaintPlus, NoamOpt
from torch.utils.data import DataLoader
from data_processing import * 
from dataloader import *
from sklearn.metrics import roc_auc_score
from config import Config
import os


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_layers = Config.num_layers
    num_heads = Config.num_heads
    d_model = Config.model_dim
    d_ffn = d_model*4
    max_len = Config.max_len
    n_questions = Config.num_question
    n_tasks = Config.num_test

    seq_len = Config.seq_len
    dropout = Config.dropout
    epochs = Config.epochs
    batch_size = Config.batch_size
    lr = Config.learning_rate

    train = pd.read_csv(Config.train_path)
    test = pd.read_csv(Config.test_path)
    train = get_time_lag(train)
    train = get_elapsed_time(train)
    test = get_time_lag(test)
    test = get_elapsed_time(test)
    train_group, val_group = data_processing(train, test) 
    
    sub_val_seq = DataToSequenceSub(val_group,seq_len)
    sub_val_size = len(sub_val_seq)
    sub_val_loader = DataLoader(sub_val_seq, batch_size=batch_size, shuffle=False, num_workers=8)

    model = SaintPlus(seq_len=seq_len, num_layers=num_layers, d_ffn=d_ffn, d_model=d_model, num_heads=num_heads,
                      max_len=max_len, n_questions=n_questions, n_tasks=n_tasks, dropout=dropout)

    checkpoint = torch.load('/opt/ml/input/DKT/code/saint+/model/best_saint_withitemavg.pt', map_location=device)
    state_dict = checkpoint['net']
    model.load_state_dict(state_dict)

    model.to(device)


    model.eval()
    submission = []
    for step, data in enumerate(sub_val_loader):
        content_ids = data[0].to(device).long()
        time_lag = data[1].to(device).float()
        ques_elapsed_time = data[2].to(device).float()
        item_aver = data[3].to(device).float()
        answer_correct = data[4].to(device).long()
        label = data[5].to(device).float()
        preds = model(content_ids, time_lag, ques_elapsed_time, item_aver, answer_correct)
        preds1 = preds[:, -1]
        submission.extend(preds1.data.cpu().numpy())

    
    print(len(submission))
    submission3 = pd.DataFrame()
    submission3['id'] = np.arange(744)
    submission3['prediction'] = submission
    submission3.to_csv("DKT/code/saint+/sub/submissionSatin_withavgitem.csv", index=False)
    
main()