from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import copy
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')

# 单独训练
class Statistical_Experiment(Exp_Basic):
    def __init__(self, args):
        super(Statistical_Experiment, self).__init__(args)
        self.args = args
        self.imp_model = self._bulid_imputation_model()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _bulid_imputation_model(self):
        imp_args = copy.deepcopy(self.args)
        imp_args.task_name = 'imputation'
        imp_args.label_len = 0
        imp_args.pred_len = 0
        print(vars(self.args))
        ###Crossformer的ETTh1,ETTm1,weather数据填补的下面三个参数和其他的填补模型不一样
        if (self.args.model == 'Crossformer' and self.args.data_path == 'weather.csv'):
            imp_args.d_model = 64
            imp_args.d_ff = 64
            imp_args.top_k = 3
        elif (self.args.model == 'Crossformer' and self.args.data_path == 'ETTm1.csv'):
            imp_args.d_model = 64
            imp_args.d_ff = 64
            imp_args.top_k = 3
        elif (self.args.model == 'Crossformer' and self.args.data_path == 'ETTh1.csv'):
            imp_args.d_model = 64
            imp_args.d_ff = 64
            imp_args.top_k = 3
        else:
            imp_args.d_model = 128
            imp_args.d_ff = 128
            imp_args.top_k = 5
        ##下面三行是为PatchTST新加的
        imp_args.e_layers = 2
        imp_args.batch_size = 16
        imp_args.learning_rate = 0.001
        print(vars(imp_args))
        imp_model = self.model_dict[self.args.model].Model(imp_args)

        # 装载填补模型权重
        imp_model.load_state_dict(torch.load(self.args.imp_model_pt), strict=False)
        #imp_model.load_state_dict(torch.load(self.args.imp_model_pt, map_location=torch.device('cpu')), strict=False)
        imp_model.to("cuda")
        imp_model.eval()
        return imp_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

    # 计算填补误差 (MAE 和 MSE)
    def calc_mae_mse(self, original, filled, mask):
        mae_fn = nn.L1Loss(reduction='none')  # 计算每个位置的 MAE，不做均值化
        mse_fn = nn.MSELoss(reduction='none')  # 计算每个位置的 MSE，不做均值化

        # 计算原始和填充数据的 MAE 和 MSE
        mae = mae_fn(filled, original)  # 计算每个位置的 MAE
        mse = mse_fn(filled, original)  # 计算每个位置的 MSE

        # 检查维度

        # 只对被掩盖的部分进行误差统计
        mae = (mae * (1 - mask)).mean()  # 被掩盖部分的 MAE
        mse = (mse * (1 - mask)).mean()  # 被掩盖部分的 MSE

        return mae, mse


    def calc_interval_mae_mse(self, original, filled, mask, time_mask):
        mae_results = {}
        mse_results = {}

        B, T, N = original.shape

        # 遍历所有样本和特征维度
        for i in range(B):
            for j in range(N):
                # 找到掩码位置为 0 的时间点
                masked_time_steps = (time_mask[i] == 0)

                # 找到连续掩码的时间段
                intervals = []
                count = 0
                for t in range(T):
                    if masked_time_steps[t]:
                        count += 1  # 连续掩码计数
                    elif count > 0:  # 当遇到非掩码点时，记录当前的连续掩码长度
                        intervals.append(count)
                        count = 0
                if count > 0:  # 处理结尾的连续掩码
                    intervals.append(count)
                
                # 遍历每个间隔（掩码长度），分别计算 MAE 和 MSE
                for interval in intervals:
                    # 如果间隔长度不存在字典中，则初始化
                    if interval not in mae_results:
                        mae_results[interval] = []
                        mse_results[interval] = []

                    # 获取掩码的位置
                    interval_mask = mask[i, :, j]

                    # 计算当前间隔上的 MAE 和 MSE
                    mae, mse = self.calc_mae_mse(original[i, :, j], filled[i, :, j], interval_mask)

                    # 保存结果
                    mae_results[interval].append(mae.item())
                    mse_results[interval].append(mse.item())

        # 计算不同间隔的平均 MAE 和 MSE
        interval_mae_avg = {interval: sum(maes) / len(maes) for interval, maes in mae_results.items()}
        interval_mse_avg = {interval: sum(mses) / len(mses) for interval, mses in mse_results.items()}

        # Sort results by keys
        interval_mae_avg = dict(sorted(interval_mae_avg.items()))
        interval_mse_avg = dict(sorted(interval_mse_avg.items()))

        return interval_mae_avg, interval_mse_avg


  
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')

        imp_mse = []
        imp_mae = []
        
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            mse_fn = nn.MSELoss()
            mae_fn = nn.L1Loss()
            for i, (batch_x_raw, batch_y_raw, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x_raw = batch_x_raw.float().to(self.device)
                batch_y_raw = batch_y_raw.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                ## 填补
                ########################################
                ##  复制一个batch_x用于后续计算填补损失
                x_raw = batch_x_raw.clone().detach().cpu()

                # 假设 batch_x_raw 是形状为 (B, T, N) 的输入数据
                B, T, N = batch_x_raw.shape

                # 生成时间步的掩码，保证所有通道在相同的时间点上掩盖
                time_mask = torch.rand((B, T)).to(self.device)
                time_mask[time_mask <= self.args.mask_rate] = 0  # masked
                time_mask[time_mask > self.args.mask_rate] = 1  # remained

                # 将时间掩码应用于所有通道 (N)，扩展维度，确保同一时间步的所有特征使用相同的掩码
                mask = time_mask.unsqueeze(-1).expand(-1, -1, N)

                # 将掩盖的部分置为 0
                inp = batch_x_raw.masked_fill(mask == 0, 0)

                # 输入填补模型
                batch_x_imp = self.imp_model(inp, batch_x_mark, None, None, mask)

                # 补回去被填充的部分
                batch_x_imp = batch_x_raw * mask + batch_x_imp * (1 - mask)

            # 创建 MAE 和 MSE 计算函数
            mae_fn = nn.L1Loss(reduction='none')  # 计算每个位置的 MAE，不做均值化
            mse_fn = nn.MSELoss(reduction='none')  # 计算每个位置的 MSE，不做均值化

            # 计算 MAE 和 MSE 的平均值，按不同掩码间隔
            interval_mae_avg, interval_mse_avg = self.calc_interval_mae_mse(batch_x_raw, batch_x_imp, mask, time_mask)

            # 输出统计结果
            print("MAE by Interval:", interval_mae_avg)
            print("MSE by Interval:", interval_mse_avg)



                

        # 保存结果
        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999

        # 保存 MAE 和 MSE 的平均值按不同间隔
        result_file = "interval_mae_mse_results.txt"

        with open(result_file, 'a') as f:
            f.write(setting + "  \n")

            f.write("Interval MSE Averages:\n")
            for interval, mse_avg in interval_mse_avg.items():
                f.write(f"Interval {interval}: MSE = {mse_avg}\n")

            f.write("Interval MAE Averages:\n")
            for interval, mae_avg in interval_mae_avg.items():
                f.write(f"Interval {interval}: MAE = {mae_avg}\n")
            
            f.write('\n')
            f.close()

        print(f"Results saved in {result_file}")

        return
