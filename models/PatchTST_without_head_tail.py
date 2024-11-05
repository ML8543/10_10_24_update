import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, PatchEmbedding_mask
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    molei:对于该模块的改动集中在__init__,forecast,imputation和forward四个方法上:

    __init__方法改动:
    1)增加ReLu作为实例变量activate_fn来控制mask张量的每个元素大于等于0。
    2)对预测任务增加线性层self.projection_dm1_dm2用于对齐两个模型的d_model。

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
    2)使用Embed.py里面加的atchEmbedding_mask模块来处理mask张量。
    3)enc_out代入Encoder填补得到输出enc_out;
      再将缺失部分张量enc_out和不缺失部分的张量enc_out_complete使用mask_complete * enc_out_complete + (1 - mask_complete) * enc_out融合在一起。
    4)返回四个张量:dec_out_d_model, dec_out_time, means, stdev
    dec_out_d_model:维度(Batch_size*n_channels,patch_num,d_model),利用mask_complete融合缺失部分和不缺失部分输出张量得到
    dec_out_time:维度(B,L,N),保留一个时间序列输出来计算填补损失
    means, stdev同上forecast方法两个形参

    forward方法改动:对应上面关于形参和返回值的改动；
    增加两个可选形参:means,stdev
    imputation任务返回值改为四个;
    
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.activate_fn = nn.ReLU()
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.patch_embedding_mask = PatchEmbedding_mask(
            configs.d_model, patch_len, stride, padding, configs.dropout)

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
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
            self.d_model_fore = configs.d_model
            self.projection_dm1_dm2 = nn.Linear(configs.d_model_imp, configs.d_model, bias=True)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, means, stdev):
        # Normalization from Non-stationary Transformer
        #means = x_enc.mean(1, keepdim=True).detach()
        #x_enc = x_enc - means
        #stdev = torch.sqrt(
        #    torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #x_enc = x_enc/stdev

        # do patching and embedding
        #x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        #enc_out, n_vars = self.patch_embedding(x_enc)
        _, _, d_model_imp = x_enc.shape

        n_vars = self.n_vars
        enc_out = x_enc

        if (d_model_imp != self.d_model_fore):
            enc_out = self.projection_dm1_dm2(enc_out)
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
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
        x_enc = x_enc/stdev
        print(x_enc.shape)

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)
        print(x_enc.shape,mask.shape)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        mask_complete, _= self.patch_embedding_mask(mask)
        print(enc_out.shape,mask_complete.shape)
        enc_out_complete = enc_out

        #breakpoint()
        # 计算小于0的元素个数并打印
        num_negative = torch.sum(mask_complete < 0)
        print(num_negative.item())
        # 应用激活函数到张量,此时 mask_complete 中所有小于0的元素都会被置为0
        mask_complete = self.activate_fn(mask_complete)
        # 计算小于0的元素个数并打印
        num_negative = torch.sum(mask_complete < 0)
        print(num_negative.item())

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        print(enc_out.shape)

        dec_out_d_model = mask_complete*enc_out_complete+(1-mask_complete)*enc_out

        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        print(enc_out.shape)
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        print(enc_out.shape)


        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        print(dec_out.shape)
        dec_out = dec_out.permute(0, 2, 1)
        print(dec_out.shape)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out_time = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out_d_model, dec_out_time, means, stdev

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc/stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc/stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, means=None, stdev=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, means, stdev)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out_d_model, dec_out_time, means, stdev = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out_d_model, dec_out_time, means, stdev  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
