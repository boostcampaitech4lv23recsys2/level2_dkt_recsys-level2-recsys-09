import torch
import torch.nn as nn

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )
from torch_geometric.nn.models import LightGCN

"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, input):

        test, question, tag, _, mask, interaction = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):

        test, question, tag, _, mask, interaction = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.max_seq_len = self.args.max_seq_len
        self.intermediate_size = self.args.intermediate_size
        self.hidden_act = self.args.hidden_act
        self.drop_out = self.args.drop_out
        self.layer_norm_eps = self.args.layer_norm_eps

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        self.embedding_user = nn.Embedding(self.args.n_user + 1, self.hidden_dim // 3)  # user

        self.embedding_month = nn.Embedding(self.args.n_month + 1, self.hidden_dim // 3) # month
        self.embedding_day = nn.Embedding(self.args.n_day + 1, self.hidden_dim // 3)     # day
        self.embedding_hour = nn.Embedding(self.args.n_hour + 1, self.hidden_dim // 3)   # hour
        self.embedding_minute = nn.Embedding(self.args.n_minute + 1, self.hidden_dim // 3)   # minute
        self.embedding_second = nn.Embedding(self.args.n_second + 1, self.hidden_dim // 3)   # second

        #self.linear_stime = nn.Linear(1, self.hidden_dim // 3)   # stime

        # get embedding from LightGCN
        lightgcn = LightGCN(7442+9454, embedding_dim=64, num_layers=6)
        lightgcn.load_state_dict(torch.load("/opt/ml/workspace/kch_dkt/code/lightgcn/weight/best_model.pt")["model"])
        edge = lightgcn.get_embedding(torch.randint(7442+9454, size=(2, 1))).detach().clone()
        self.lightgcn_embedding = nn.Embedding.from_pretrained(torch.cat((edge, torch.rand((2,64)),)), freeze=False)
        """
        self.lightgcn_embedding = self.lightgcn.embedding.weight[7442:].detach().clone()
        lightgcn_embedding = torch.cat((lightgcn_embedding,torch.rand((2,64)),))
        self.LGCN_embedding_question = nn.Embedding.from_pretrained(lightgcn_embedding, freeze=False)
        """

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 10 + 64 * 2, self.hidden_dim)

        # Bert config
        self.config = BertConfig(
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=self.max_seq_len,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.drop_out,  # 0.1
            attention_probs_dropout_prob=self.drop_out,   # 0.1
            layer_norm_eps=self.layer_norm_eps,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Bert_pretrined
        #self.encoder_pretrained = BertModel.from_pretrained("bert-base-uncased")    # input dim: 768

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        test, question, tag, _, user, month, day, hour, minute, second, mask, interaction = input   # stime
        batch_size = interaction.size(0)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_user = self.embedding_user(user)  # user

        embed_month = self.embedding_month(month)   # month
        embed_day = self.embedding_day(day)     # day
        embed_hour = self.embedding_hour(hour)  # hour
        embed_minute = self.embedding_minute(minute)    # minute
        embed_second = self.embedding_second(second)    # second

        # stime: [256, seqlen] -> [256, seqlen, hdim]
        #unsqueezed_stime = stime.unsqueeze(2)   # [256, seqlen, 1]
        #embed_stime = self.linear_stime(unsqueezed_stime)   # [256, seqlen, hdim]

        # [256, seqlen, 64]
        lgcn_embed_user = self.lightgcn_embedding(user) # lgcn user
        lgcn_embed_question = self.lightgcn_embedding(question) # lgcn question

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_user,    # user

                embed_month,    # month
                embed_day,  # day
                embed_hour, # hour
                embed_minute,   # minute
                embed_second,   # second

                #embed_stime,    # stime

                lgcn_embed_user, # lgcn user
                lgcn_embed_question # lgcn question
            ],
            2,
        )

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers.last_hidden_state # encoded_layers[0]

        # Bert_pretrained
        #encoded_layers = self.encoder_pretrained(inputs_embeds=X, attention_mask=mask)
        #out = encoded_layers.last_hidden_state

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out).view(batch_size, -1)
        return out
