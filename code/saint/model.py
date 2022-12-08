import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Reference:
https://arxiv.org/abs/2002.07033
"""

class FFN(nn.Module):
    def __init__(self, d_ffn, d_model, dropout):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ffn) #[batch, seq_len, ffn_dim]
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(d_ffn, d_model) #[batch, seq_len, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        return self.dropout(x)

class SaintPlus(nn.Module):
    def __init__(self, seq_len, num_layers, d_ffn, d_model, num_heads, max_len, n_questions, n_tasks, dropout):
        super(SaintPlus, self).__init__()
        self.d_model = d_model
        self.n_questions = n_questions
        self.num_heads = num_heads

        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.contentId_emb = nn.Embedding(n_questions+1, d_model)
        #self.task_emb = nn.Embedding(n_tasks+1, d_model)
        
        self.timelag_emb = nn.Linear(1, d_model, bias=False)
        self.elapsedT_emb = nn.Linear(1, d_model, bias=False)
        self.itemAver_emb = nn.Linear(1, d_model, bias=False)
        self.userAver_emb = nn.Linear(1, d_model, bias=False)
        self.tagAver_emb =nn.Linear(1, d_model, bias=False)
        
        self.answerCorr_emb = nn.Embedding(3, d_model)

        self.emb_dense1 = nn.Linear(2*d_model, d_model)
        self.emb_dense2 = nn.Linear(6*d_model, d_model)

        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=d_ffn, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.FFN = FFN(d_ffn, d_model, dropout=dropout)
        self.final_layer = nn.Linear(d_model, 1)

    def forward(self, content_ids,time_lag, ques_elapsed_time,  item_aver, user_aver ,tag_aver,answer_correct):
        device = content_ids.device
        seq_len = content_ids.shape[1]

        content_id_emb = self.contentId_emb(content_ids)
        #task_id_emb = self.task_emb(test_id)
        time_lag = torch.log(time_lag+1)
        time_lag = time_lag.view(-1, 1) # [batch*seq_len, 1]
        time_lag = self.timelag_emb(time_lag) # [batch*seq_len, d_model]
        time_lag = time_lag.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]
        elapsed_time = torch.log(ques_elapsed_time+1)
        elapsed_time = elapsed_time.view(-1, 1) # [batch*seq_len, 1]
        elapsed_time = self.elapsedT_emb(elapsed_time) # [batch*seq_len, d_model]
        elapsed_time = elapsed_time.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]
        ####
        item_aver = torch.log(item_aver+1)
        item_aver = item_aver.view(-1, 1) # [batch*seq_len, 1]
        item_aver = self.itemAver_emb(item_aver) # [batch*seq_len, d_model]
        item_aver = item_aver.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]

        user_aver = torch.log(user_aver+1)
        user_aver = user_aver.view(-1, 1) # [batch*seq_len, 1]
        user_aver = self.userAver_emb(user_aver) # [batch*seq_len, d_model]
        user_aver = user_aver.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]
        
        tag_aver = torch.log(tag_aver+1)
        tag_aver = tag_aver.view(-1, 1) # [batch*seq_len, 1]
        tag_aver = self.tagAver_emb(tag_aver) # [batch*seq_len, d_model]
        tag_aver = tag_aver.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]

        answer_correct_emb = self.answerCorr_emb(answer_correct)

        encoder_val = torch.cat((content_id_emb ,time_lag), axis=-1)
        encoder_val = self.emb_dense1(encoder_val)
        decoder_val = torch.cat((time_lag, elapsed_time, item_aver, user_aver ,tag_aver,answer_correct_emb), axis=-1)
        decoder_val = self.emb_dense2(decoder_val)

    
        pos = torch.arange(seq_len).unsqueeze(0).to(device)
        pos_emb = self.pos_emb(pos)
        encoder_val += pos_emb
        decoder_val += pos_emb

        over_head_mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool'))
        over_head_mask = over_head_mask.to(device)

        encoder_val = encoder_val.permute(1, 0, 2)
        decoder_val = decoder_val.permute(1, 0, 2)
        decoder_val = self.transformer(encoder_val, decoder_val, src_mask=over_head_mask, tgt_mask=over_head_mask, memory_mask=over_head_mask)

        decoder_val = self.layer_norm(decoder_val)
        decoder_val = decoder_val.permute(1, 0, 2)

        final_out = self.FFN(decoder_val)
        final_out = self.layer_norm(final_out + decoder_val)
        final_out = self.final_layer(final_out)
        final_out = torch.sigmoid(final_out)
        return final_out.squeeze(-1)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))