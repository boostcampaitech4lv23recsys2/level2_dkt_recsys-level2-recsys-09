import os
from config import CFG
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn.models import LightGCN
from typing import Optional, Union

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList, Linear
from torch.nn.modules.loss import _Loss
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor

def build(n_node, weight=None, logger=None, **kwargs):
    model=LGCN_v2(n_node,
        embedding_dim=CFG.embedding_dim,
        num_layers=CFG.num_layers,
        alpha=CFG.alpha,
        logger=logger.getChild("build"),
        **CFG.build_kwargs)
    
    if weight:
        if not os.path.isfile(weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def train(
    model,
    train_data,
    valid_data,
    n_epoch=100,
    learning_rate=0.01,
    use_wandb=False,
    weight=None,
    logger=None,
):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists(weight):
        os.makedirs(weight)

    # if valid_data is None:
    #     eids = np.arange(len(train_data["label"]))
    #     eids = np.random.permutation(eids)[:1000] #트레인 데이터들중 랜덤인덱스느낌 앞에 1000개만
    #     edge, label = train_data["edge"], train_data["label"]
    #     label = label.to("cpu").detach().numpy()
    #     valid_data = dict(edge=edge[:, eids], label=label[eids])
    
    # edge, label = valid_data["edge"], valid_data["label"]
    # label = label.to("cpu").detach().numpy()
    # valid_data = dict(edge=edge, label=label)
    label = valid_data["label"]
    label = label.to("cpu").detach().numpy()
        

    logger.info(f"Training Started : n_epoch={n_epoch}")
    best_auc, best_epoch = 0, -1
    for e in range(n_epoch):
        # forward
        pred = model(train_data)
        loss = model.link_pred_loss(pred, train_data)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("val:",valid_data)
        with torch.no_grad():
            prob = model.predict_link(valid_data, prob=True)
            
            prob = prob.detach().cpu().numpy()
            labels=valid_data["label"].cpu().numpy()
            acc = accuracy_score(labels, prob > 0.5)
            auc = roc_auc_score(labels, prob)
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


def inference(model, data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(data, prob=True)
        return pred

class LGCN_v2(LightGCN):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__(num_nodes,embedding_dim,num_layers)

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.n_user =CFG.n_user
        self.n_item =CFG.n_item
        self.n_test =CFG.n_test
        self.n_tag =CFG.n_tag
        self.device = CFG.device
        
        self.user_embedding = Embedding(self.n_user+1, embedding_dim)
        self.item_embedding = Embedding(self.n_item+1, embedding_dim)
        self.test_embedding = Embedding(self.n_test+1, embedding_dim)
        self.tag_embedding = Embedding(self.n_tag+1, embedding_dim)
        
        self.finalitems = Linear(embedding_dim*3,embedding_dim)
        
        self.user_item = Linear(embedding_dim,embedding_dim)
        
        
        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.embedding = Embedding(num_nodes, embedding_dim)
        
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, input):
        #edge_index와 edge_label_index,input['edge'] 같은값임
        #print(input)
        #print(input['item'])
        
        edges=input['edge']
        item=input['item']
        test=input['test']
        tag=input['tag']
        
        emb_item=self.item_embedding(item)
        emb_test=self.test_embedding(test)
        emb_tag=self.tag_embedding(tag)
        
        itemcat=torch.cat([emb_item,emb_test,emb_tag],1).to(self.device)
        
        finalemb_items=self.finalitems(itemcat)
        
        user_item_emb=torch.cat([self.user_embedding.weight, finalemb_items],0).to(self.device)
        
        x = user_item_emb
        out = x * self.alpha[0]
        
        # x = self.embedding.weight
        # out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edges)
            out = out + x * self.alpha[i + 1]
            
        out_src = out[edges[0]]
        out_dst = out[edges[1]]
        
        return (out_src * out_dst).sum(dim=-1)
    
    def predict_link(self, input,
                     prob: bool = False) -> Tensor:
        pred = self(input).sigmoid()
        return pred if prob else pred.round()


    def link_pred_loss(self, pred: Tensor, input,
                       **kwargs) -> Tensor:
        edge_label=input['label']
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')

