import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
            ##下面三行是三种尝试去除线性投影层的方法
            #下面第一行这个虽然线性层输出维度不变，但还是会经过线性投影
            #self.projection = nn.Linear(configs.d_model, configs.d_model, bias=False)
            self.projection_remain = nn.Identity()
            #self.projection_remain = nn.Lambda(lambda x: x)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, means, stdev):
        _, _, N = means.shape

        print(means)
        print(stdev)

        # Embedding
        #enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = x_enc
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        print(enc_out.shape)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        print(dec_out.shape)
        # De-Normalization from Non-stationary Transformer
        #dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        print(means)
        print(stdev)
        print(means.shape)
        print(stdev.shape)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer without mask to balance
        #means = x_enc.mean(1, keepdim=True).detach()
        #x_enc = x_enc - means
        #stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #x_enc /= stdev

        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        
        _, L, N = x_enc.shape
        print(x_enc.shape)

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        print(enc_out.shape)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        print(enc_out.shape)
        
        dec_out_d_model = enc_out

        _, L_d_model, N_d_model = dec_out_d_model.shape

        dec_out_d_model = self.projection_remain(dec_out_d_model)
        #dec_out_d_model = self.projection_remain(dec_out_d_model).permute(0, 2, 1)[:, :, :N]
        dec_out_time = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]


        # De-Normalization from Non-stationary Transformer
        print(x_enc.shape)
        print(dec_out_d_model.shape)
        print(dec_out_time.shape)
        print(stdev.shape)


        #dec_out_d_model = dec_out_d_model * (stdev[:, 0, :].unsqueeze(1).repeat(1, L_d_model, 1))
        #dec_out_d_model = dec_out_d_model + (means[:, 0, :].unsqueeze(1).repeat(1, L_d_model, 1))

        # De-Normalization from Non-stationary Transformer
        dec_out_time = dec_out_time * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out_time = dec_out_time + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out_d_model, dec_out_time, means, stdev

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, means=None, stdev=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, means, stdev)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out_d_model, dec_out_time, means, stdev = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out_d_model, dec_out_time, means, stdev  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
