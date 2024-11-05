import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    molei:对于该模块的改动集中在__init__,forecast,imputation和forward四个方法上:

    __init__方法改动:
    1)增加ReLu作为实例变量activate_fn来控制mask张量的每个元素大于等于0。
    2)对预测任务增加线性层self.projection_dm1_dm2用于对齐两个模型的d_model。
    3)对填补任务增加线性层self.projection_mask用于将mask张量seq_len投影到d_model维。

    forecast方法改动:
    1)增加两个形参
    param:means:填补方法最开始标准化时计算的均值;
    param:stdev:填补方法最开始标准化时计算的标准差；
    2)去掉了一开始的计算均值和标准差,使用两个传入的实参代替。
    3)去掉embedding,直接将填补的输出输入到预测的Encoder中。
    4)增加对其两个模型的d_model对齐的线性层。
    5)在预测模型的最后使用传入的means和stdev来做逆归一化。

    imputation方法改动:
    1)真实数据(含缺失)代入embedding得到enc_out赋给enc_out_complete作为不缺失部分的向量。
    2)enc_out代入Encoder填补得到输出enc_out,
    3)将缺失部分张量enc_out和不缺失部分的张量enc_out_complete使用mask_complete * enc_out_complete + (1 - mask_complete) * enc_out融合在一起。
    4)返回四个张量:dec_out_d_model, dec_out_time, means, stdev
    dec_out_d_model:维度(B,N+x_mark,d_model),利用mask_complete融合缺失部分和不缺失部分输出张量得到
    dec_out_time:维度(B,L,N),保留一个时间序列输出来计算填补损失
    means, stdev同上forecast方法两个形参

    forward方法改动:对应上面关于形参和返回值的改动；
    增加两个可选形参:means,stdev
    imputation任务返回值改为四个;
    
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.activate_fn = nn.ReLU()
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
            self.projection_dm1_dm2 = nn.Linear(configs.d_model_imp, configs.d_model, bias=True)
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            self.d_model_fore = configs.d_model
        if self.task_name == 'imputation':
            #用于将mask张量seq_len投影到d_model维
            self.projection_mask = nn.Linear(configs.seq_len, configs.d_model, bias=True)
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
            ##下面三行是三种尝试去除线性投影层的方法
            #下面第一行这个虽然线性层输出维度不变，但还是会经过线性投影
            #self.projection = nn.Linear(configs.d_model, configs.d_model, bias=False)
            #self.projection_remain = nn.Identity()
            #self.projection_remain = nn.Lambda(lambda x: x)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, means, stdev):
        _, _, N = means.shape

        _, _, d_model_imp = x_enc.shape
        #print(means)
        #print(stdev)
        # Embedding
        #enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = x_enc
        #breakpoint()
        print(d_model_imp)
        print(self.d_model_fore)
        #breakpoint()
        print(x_enc.device, x_mark_enc.device, x_dec.device, x_mark_dec.device, means.device, stdev.device)
        if (d_model_imp != self.d_model_fore):
            enc_out = self.projection_dm1_dm2(enc_out) 
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        print(enc_out.shape)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        print(dec_out.shape)
        # De-Normalization from Non-stationary Transformer
        #dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #print(means)
        #print(stdev)
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
        B, L, N = x_enc.shape
        print(x_enc.shape)

        print(mask.shape)
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        #留一个enc_out_complete(嵌入结果不带入填补Encoder)用于后面将缺失部分和不缺失部分的向量连在一起
        enc_out_complete = enc_out
        print(enc_out.shape)
        _, N_x_mark, d_model = enc_out.shape
        # 计算需要填充的维度
        padding_size = N_x_mark - N
        print(padding_size)
        # 创建一个填充为 1 的张量，维度 (a, b, padding_size)
        padding_tensor = torch.ones(B, L, padding_size)
        print(padding_tensor.shape)
        #用于取不缺失数据的嵌入向量部分的mask矩阵，其在通道维度上多个x_mark
        print(mask.shape)
        #检查 mask 和 padding_tensor 张量所在设备
        print(f"mask 所在的设备: {mask.device}")
        print(f"padding_tensor 所在的设备: {padding_tensor.device}")
        padding_tensor = padding_tensor.to(mask.device)  # 将 padding_tensor 移动到与 mask 相同的设备
        mask_complete = torch.cat((mask, padding_tensor), dim=2)
        print(mask_complete.shape)
        print(mask_complete.shape)  # 检查 mask_complete 的形状

        print(self.projection_mask.weight.shape)  # 检查投影层的权重形状
        ##############
        #mask_complete = self.projection_mask(mask_complete.view(-1, L)).view(B, d_model, N_x_mark)
        print(mask_complete.shape)
        mask_complete = mask_complete.transpose(1, 2)
        mask_complete = self.projection_mask(mask_complete)
        print(mask_complete.shape)
        print(mask_complete)
        #breakpoint()
        # 计算小于0的元素个数并打印
        num_negative = torch.sum(mask_complete < 0)
        print(num_negative.item())
        # 应用激活函数到张量,此时 mask_complete 中所有小于0的元素都会被置为0
        mask_complete = self.activate_fn(mask_complete)
        # 计算小于0的元素个数并打印
        num_negative = torch.sum(mask_complete < 0)
        print(num_negative.item())

        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        print(enc_out.shape)
        
        #dec_out_d_model = enc_out
        #上面的dec_out_d_model是不考虑缺失部分和不缺失部分的输出向量的融合的，下面的考虑了融合：mask×real_feature+(1-mask)×missing_feature
        dec_out_d_model = mask_complete * enc_out_complete + (1 - mask_complete) * enc_out

        _, L_d_model, N_d_model = dec_out_d_model.shape

        #dec_out_d_model = self.projection_remain(dec_out_d_model)
        dec_out_time = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]


        # De-Normalization from Non-stationary Transformer
        print(x_enc.shape)
        print(dec_out_d_model.shape)
        print(dec_out_time.shape)
        print(stdev.shape)


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
            print(x_enc.device, x_mark_enc.device, x_dec.device, x_mark_dec.device, means.device, stdev.device)
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
