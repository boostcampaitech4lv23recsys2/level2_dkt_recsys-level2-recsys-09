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
from args import parser
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거


def main():
    device = "cpu" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()

    d_ffn = args.model_dim*4

    train = pd.read_csv(Config.train_path)
    test = pd.read_csv(Config.test_path)
    train = get_time_lag(train)
    train = get_elapsed_time(train)
    test = get_time_lag(test)
    test = get_elapsed_time(test)
    train_group, val_group, test_group = data_processing(train, test) 
    sub_val_seq = DataToSequenceSub(test_group,args.seq_len)
    sub_val_size = len(sub_val_seq)
    sub_val_loader = DataLoader(sub_val_seq, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = SaintPlus(seq_len=args.seq_len, num_layers=args.num_layers, d_ffn=d_ffn, d_model=args.model_dim, num_heads=args.num_heads,
                    max_len=args.max_len, n_questions=args.num_question, n_tasks=args.num_test, dropout=args.dropout)

    checkpoint = torch.load('/opt/ml/input/DKT/code/saint+/model/REALBEST_EMB.pt', map_location=device)
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
        user_aver = data[4].to(device).float()
        tag_aver = data[5].to(device).float()
        answer_correct = data[6].to(device).long()
        label = data[7].to(device).float()

        preds = model(content_ids,time_lag, ques_elapsed_time, item_aver, user_aver,tag_aver,answer_correct)
        preds1 = preds[:, -1]
        submission.extend(preds1.data.cpu().numpy())

    
    real_sub = pd.DataFrame()
    real_sub['id'] = np.arange(744)
    real_sub['prediction'] = submission
    real_sub.to_csv("DKT/code/saint+/sub/REALBEST_EMB.csv", index=False)
    
if __name__ == "__main__":
    main()