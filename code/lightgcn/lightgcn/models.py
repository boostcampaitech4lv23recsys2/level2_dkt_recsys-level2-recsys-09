import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn.models import LightGCN


# 예측을 위한 모델을 구성하고 불러오기 ~ train이후에 
def build(n_node, weight=None, logger=None, **kwargs):
    model = LightGCN(n_node, **kwargs)
    # best weigh가 저장되어있는 것을 가져와 확인
    if weight:
        if not os.path.isfile(weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        # weight 지정된 모델 반환
        return model

# 모델 학습 과정
def train(
    model,
    train_data,
    valid_data=None,
    n_epoch=100,
    learning_rate=0.01,
    use_wandb=False,
    weight=None,
    logger=None,
):
    model.train()
    # optimizer Adam으로 지정 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # weight 폴더 만들어주는 것부터 시작
    if not os.path.exists(weight):
        os.makedirs(weight)

    # valid가 따로 지정되어 있지 않기에 자체적으로 valid 만들기 ~ 이미 데이터 상에서 id와 시간순으로 sort를 해놨지에 id별 valid가 가능
    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids])

    # epoch만큼 학습진행
    logger.info(f"Training Started : n_epoch={n_epoch}")
    best_auc, best_epoch = 0, -1
    for e in range(n_epoch):
        # forward 부분 ~ 노드를 전달하면 예측
        pred = model(train_data["edge"])
        # 예측값과 label값을 비교
        loss = model.link_pred_loss(pred, train_data["label"][-1])

        # backward 부분
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 점수 확인 부분
        with torch.no_grad():
            # valid 값에 대해 예측 후 acc와 auc 점수 추출 ~ 최고점만 저장하기
            prob = model.predict_link(valid_data["edge"], prob=True)
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(valid_data["label"], prob > 0.5)
            auc = roc_auc_score(valid_data["label"], prob)
            logger.info(
                f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}"
            )
            if use_wandb:
                import wandb

                wandb.log(dict(loss=loss, acc=acc, auc=auc))

        if weight:
            if auc > best_auc:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Best AUC"
                )
                best_auc, best_epoch = auc, e
                torch.save(
                    {"model": model.state_dict(), "epoch": e + 1},
                    os.path.join(weight, f"best_model.pt"),
                )
    torch.save(
        {"model": model.state_dict(), "epoch": e + 1},
        os.path.join(weight, f"last_model.pt"),
    )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")

# test 제출을 위한 predict
def inference(model, data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(data["edge"], prob=True)
        return pred
