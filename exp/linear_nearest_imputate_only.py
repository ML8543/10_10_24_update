from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
from scipy.interpolate import interp1d
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')

# 对mask后的tensor进行插值。interp1d()函数的kind参数决定了插值方法：nearest(最近邻)、linear(线性)
def interpolate(tensor,device,kind):
    tensor = tensor.cpu()
    B, T, N = tensor.shape
    filled_tensor = torch.zeros_like(tensor)
    
    for b in range(B):
        for n in range(N):
            data = tensor[b, :, n].numpy()
            mask = data != 0
            if np.any(mask):
                indices = np.arange(T)
                interp_func = interp1d(indices[mask], data[mask], bounds_error=False, fill_value="extrapolate",kind=kind)
                filled_tensor[b, :, n] = torch.tensor(interp_func(indices))
            else:
                filled_tensor[b, :, n] = torch.tensor(data)  # No non-zero data to interpolate

    return filled_tensor.to(device)

class Exp_Long_Term_Forecast_Imp_R_Linear_Nearest_Only(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Imp_R_Linear_Nearest_Only, self).__init__(args)


    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

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
        

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        imp_mse = []
        imp_mae = []
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            mse_fn = nn.MSELoss()
            mae_fn = nn.L1Loss()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                ########################################
                ##  复制一个batch_x用于后续计算填补损失
                x_raw = batch_x.clone().detach().cpu()

                ## random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                batch_x = batch_x.masked_fill(mask == 0, 0)
                if self.args.interpolate != 'no':
                    batch_x = interpolate(batch_x,self.device,self.args.interpolate)

                # 复制一个用于后续损失计算
                x_imp = batch_x.clone().detach().cpu()
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                
                # Move tensors to CPU before indexing
                batch_x = batch_x.detach().cpu()
                x_imp = x_imp.detach().cpu()

                # Calculate masked indices on CPU
                mask = mask.to('cpu')

                # Index tensors using CPU indices
                imp_mae.append(mae_fn(x_raw[mask == 0], x_imp[mask == 0]).item())
                imp_mse.append(mse_fn(x_raw[mask == 0], x_imp[mask == 0]).item())
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                #if i % 20 == 0:
                #    input = batch_x.detach().cpu().numpy()
                #    if test_data.scale and self.args.inverse:
                #        shape = input.shape
                #        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        imp_mae = np.mean(imp_mae)
        imp_mse = np.mean(imp_mse)

        
        # result save
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
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('imp_mse:{}, imp_mae:{}'.format(imp_mse, imp_mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('(use {} interpolate method to imputate data after mask)\n'.format(self.args.interpolate))
        f.write('imp_mse:{}, imp_mae:{}'.format(imp_mse, imp_mae))
        f.write('\n')
        f.write('\n')
        f.close()

        #np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        #np.save(folder_path + 'pred.npy', preds)
        #np.save(folder_path + 'true.npy', trues)

        return
