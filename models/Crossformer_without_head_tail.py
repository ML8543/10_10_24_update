import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding, PatchEmbedding_mask
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from models.PatchTST import FlattenHead


from math import ceil


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = configs.task_name

        # The padding operation to handle invisible sgemnet length
        #首先计算输入序列的填充长度,ceil表示向上取整
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        #计算输出序列的填充长度
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        #这行代码计算输入序列中包含的段数。它将填充后的输入序列长度（self.pad_in_len）除以段长度（self.seg_len），然后向下取整
        #在Python中，当使用整型除法时，结果会自动向下取整。//是整型除法
        self.in_seg_num = self.pad_in_len // self.seg_len
        #这行代码计算输出序列中包含的段数。它首先将输入序列中的段数（self.in_seg_num）除以窗口大小（self.win_size）的(configs.e_layers - 1)次方，然后向上取整
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        #这行代码计算每个头的特征数量。它将模型中每个头的特征数量（configs.d_model）乘以输出序列中的段数（self.out_seg_num）
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0)
        self.enc_value_embedding_mask = PatchEmbedding_mask(configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0)

        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if l is 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
                            1, configs.dropout,
                            self.in_seg_num if l is 0 else ceil(self.in_seg_num / self.win_size ** l), configs.factor
                            ) for l in range(configs.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), configs.factor, configs.d_model, configs.n_heads,
                                           configs.d_ff, configs.dropout),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    self.seg_len,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    # activation=configs.activation,
                )
                for l in range(configs.e_layers + 1)
            ],
        )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)
        elif self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection_dm1_dm2 = nn.Linear(configs.d_model_imp, configs.d_model, bias=True)
            self.d_model_fore = configs.d_model



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, means, stdev):
        # embedding
        #x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        #x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars)
        #x_enc += self.enc_pos_embedding
        #x_enc = self.pre_norm(x_enc)
        _, _, _, d_model_imp = x_enc.shape
        print(x_enc.shape)
        print(d_model_imp, self.d_model_fore)
        enc_out = x_enc
        if (d_model_imp != self.d_model_fore):
            enc_out = self.projection_dm1_dm2(x_enc)
        print(enc_out.shape)
        enc_out, attns = self.encoder(enc_out)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        mask_complete, _ = self.enc_value_embedding_mask(mask.permute(0, 2, 1))
        
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        \
        mask_complete = rearrange(mask_complete, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out_complete = x_enc
        enc_out, attns = self.encoder(x_enc)
        print(type(enc_out))
        enc_out_time = enc_out

        # 假设 enc_out 是一个包含多个 PyTorch 张量的列表
        for i, tensor in enumerate(enc_out):
            print(f"Shape of element {i}: {tensor.shape}")
        enc_out = enc_out[-1]
        _, _, last_layer_dim, _ = enc_out.shape
        _, _, first_layer_dim, _ = enc_out_complete.shape
        print(last_layer_dim)
        print(first_layer_dim)
        enc_out = enc_out.permute(0, 1, 3, 2)
        print(enc_out.shape)
        projection_last_first_layer = nn.Linear(last_layer_dim, first_layer_dim)
        print(f"enc_out 所在的设备: {enc_out.device}")
        print(f"权重所在的设备: {projection_last_first_layer.weight.device}")
        print(f"偏置所在的设备: {projection_last_first_layer.bias.device}")
        projection_last_first_layer = projection_last_first_layer.to(enc_out.device)
        print(f"enc_out 所在的设备: {enc_out.device}")
        print(f"权重所在的设备: {projection_last_first_layer.weight.device}")
        print(f"偏置所在的设备: {projection_last_first_layer.bias.device}")
        enc_out = projection_last_first_layer(enc_out)
        print(enc_out.shape)
        enc_out = enc_out.permute(0, 1, 3, 2)
        print(enc_out.shape)
        
        dec_out_d_model = mask_complete*enc_out_complete+(1-mask_complete)*enc_out
        
        dec_out = self.head(enc_out_time[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        print(dec_out.shape)
        
        a = stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)
        print(a.shape)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out_time = dec_out
        print(dec_out_time.shape)
    
        return dec_out_d_model, dec_out_time, means, stdev


    def anomaly_detection(self, x_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))

        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
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