import time
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model import SaintPlus, NoamOpt
from torch.utils.data import DataLoader
from data_processing import * 
from dataloader import DataToSequence
from sklearn.metrics import roc_auc_score
from config import Config
from args import parser
import os
import random
import wandb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

# 시드 고정용
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  
seed_everything(42)

# 모델 세이브
def save_model(model):
    check_point = {
        'net': model.state_dict()
    }
    torch.save(check_point,"/opt/ml/input/DKT/code/saint+/model/REALBEST_EMB.pt")
    

# 학습
def main():
    if Config.user_wandb:
        wandb.init(project="dkt-saint", entity="likesubscriberecommendai")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("현재 {} 사용중..".format(device))
    
    # 설정값 
    d_ffn = args.model_dim*4

    # 데이터 불러오고 전처리
    train = pd.read_csv(Config.train_path)
    test = pd.read_csv(Config.test_path)
    train = get_time_lag(train)
    train = get_elapsed_time(train)
    test = get_time_lag(test)
    test = get_elapsed_time(test)
    train_group, val_group,test_group = data_processing(train, test) 

    # loader에 태우고
    train_seq = DataToSequence(train_group, args.seq_len)
    train_size = len(train_seq)
    train_loader = DataLoader(train_seq, batch_size=args.batch_size, shuffle=True, num_workers=8)
    del train_seq, train_group

    val_seq = DataToSequence(val_group, args.seq_len)
    val_size = len(val_seq)
    val_loader = DataLoader(val_seq, batch_size=args.batch_size, shuffle=False, num_workers=8)
    del val_seq, val_group

    # 모델 정의
    model = SaintPlus(seq_len=args.seq_len, num_layers=args.num_layers, d_ffn=d_ffn, d_model=args.model_dim, num_heads=args.num_heads,
                    max_len=args.max_len, n_questions=args.num_question, n_tasks=args.num_test, dropout=args.dropout)
    
    if Config.user_wandb:
        wandb.watch(model)
        
    # 손실함수와 Optimizer 정의
    loss_fn = nn.BCELoss()
    optimizer = NoamOpt(args.model_dim, 1, args.warmup ,optim.Adam(model.parameters(), lr=args.learning_rate))
    #TODO learning rate scheduler 지정 ~ sweep 최고점의 오버핏을 이거로 조금 누를수 있지 않을까하는 생각 
    # 이미 NOAM에 포함
    

    # device 정의
    model.to(device)
    loss_fn.to(device)

    # 지표 저장용 리스트
    train_losses = []
    val_losses = []
    val_aucs = []
    best_auc = 0
    count=0
    
    # epoch시작
    for e in range(args.epochs):
        print("=========={}번째 EPOCH==========".format(e+1))
        model.train()
        train_loss = []
        train_labels = []
        train_preds = []
        
        for step, data in enumerate(train_loader):
            # 각 loader로 받아온 데이터를 device 지정
            content_ids = data[0].to(device).long()
            time_lag = data[1].to(device).float()
            ques_elapsed_time = data[2].to(device).float()
            item_aver = data[3].to(device).float()
            user_aver = data[4].to(device).float()
            tag_aver = data[5].to(device).float()
            answer_correct = data[6].to(device).long()
            label = data[7].to(device).float()
            optimizer.optimizer.zero_grad()

            # forward & backward
            preds = model(content_ids, time_lag, ques_elapsed_time,item_aver, user_aver ,tag_aver ,answer_correct)
           # FIXME sucess masking하는 부분 제외 ~ 
           # TODO loss backward를 각각 1과 0에 대한 마스킹으로 두번 나눠서 loss값을 두번 나누고 두번 backward해보기
            loss_mask = (answer_correct != 0)
            preds_masked = torch.masked_select(preds, loss_mask)
            label_masked = torch.masked_select(label, loss_mask)
            loss = loss_fn(preds, label)

            loss.backward()
            optimizer.step()

            # loos와 지표확인용으로 리스트에 저장
            train_loss.append(loss.item())
            train_labels.extend(label.view(-1).data.cpu().numpy())
            train_preds.extend(preds.view(-1).data.cpu().numpy())

        train_loss = np.mean(train_loss)
        train_auc = roc_auc_score(train_labels, train_preds)

        # val 부분
        model.eval()
        val_loss = []
        val_labels = []
        val_preds = []

        for step, data in enumerate(val_loader):
            content_ids = data[0].to(device).long()
            time_lag = data[1].to(device).float()
            ques_elapsed_time = data[2].to(device).float()
            item_aver = data[3].to(device).float()
            user_aver = data[4].to(device).float()
            tag_aver = data[5].to(device).float()
            answer_correct = data[6].to(device).long()
            label = data[7].to(device).float()


            preds = model(content_ids,time_lag, ques_elapsed_time,item_aver, user_aver,tag_aver,answer_correct)
            loss_mask = (answer_correct != 0)
            preds_masked = torch.masked_select(preds, loss_mask)
            label_masked = torch.masked_select(label, loss_mask)
            loss = loss_fn(preds, label)
 
            val_loss.append(loss.item())
            val_labels.extend(label.view(-1).data.cpu().numpy())
            val_preds.extend(preds.view(-1).data.cpu().numpy())

        val_loss = np.mean(val_loss)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        if Config.user_wandb:
            wandb.log(dict(loss=val_loss, auc=val_auc))

        # 최고 점수 확인용
        if val_auc > best_auc:
            print("{}번째 EPOCH에서 최고값 발생! ROCAUC : {}".format(e+1, val_auc))
            save_model(model)
            best_auc = val_auc
            # earlystop용
            count=0
        else:
            count+=1

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        print("{}번째 EPOCH 학습 종료!".format(e+1))
        print("Train Loss {:.4f}/ Val Loss {:.4f}, Val AUC {:.4f}".format(train_loss, val_loss, val_auc))
        
        
        if count==10:
            print('Early Stopping!')
            print("Train Loss {:.4f}/ Val Loss {:.4f}, Val AUC {:.4f}".format(max(train_losses), max(val_losses),max(val_aucs)))
            break
    

if __name__ == "__main__":
    seed_everything(42)
    main()