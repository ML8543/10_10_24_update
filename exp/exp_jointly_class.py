from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import copy

warnings.filterwarnings('ignore')

class joint_loss(nn.Module):
    def __init__(self, _lambda, imp_ls_fn, ds_ls_fn):
        super(joint_loss, self).__init__()
        self._lambda = _lambda
        self.imp_ls_fn = imp_ls_fn
        self.ds_ls_fn = ds_ls_fn
        
    def forward(self, x_imp, y_imp, x_ds, y_ds):
        imp_loss = self.imp_ls_fn(x_imp, y_imp)
        ds_loss = self.ds_ls_fn(x_ds, y_ds)
        total_loss = self._lambda * imp_loss + ds_loss
        return total_loss, imp_loss, ds_loss


class Exp_Jointly_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Jointly_Classification, self).__init__(args)
        self.args = args
        self.imp_model = self._build_imputation_model()

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')

        # edit begin
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        # self.args.seq_len = 256
        # edit end

        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _build_imputation_model(self):
        imp_args = copy.deepcopy(self.args)
        imp_args.task_name = 'imputation'
        imp_args.label_len = 0
        imp_args.pred_len = 0
        imp_args.d_model = 64
        imp_args.top_k = 3
        #####为了让程序不报错尝试加几行参数赋值
        ##imp_args.is_training = 1
        ##imp_args.model_id = 'ECL_mask_0.125'
        ##imp_args.model = $model_name
        ##imp_args.data = UEA
        #####
        imp_model = self.model_dict[self.args.model].Model(imp_args)
        imp_model.to(self.device)
        return imp_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        #model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam([{'params':self.model.parameters()},
                                  {'params':self.imp_model.parameters()}], lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        imp_loss_fn = nn.MSELoss()
        ds_ls_fn = nn.CrossEntropyLoss()
        criterion = joint_loss(self.args._lambda,imp_loss_fn,ds_ls_fn)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        #######################
        imp_loss_total = []
        ds_loss_total = []
        ########################
        preds = []
        trues = []
        ########################
        self.imp_model.eval()
        ########################
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                #############
                
                ########填补开始
                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)
                
                ###输入
                batch_x_imp = self.imp_model(inp, batch_x, None, None, mask)

                ###补回去被填充的部分
                batch_x_imp = batch_x*mask + batch_x_imp*(1-mask)
                
                ###复制一个用于后续损失计算
                x_imp = batch_x_imp.clone()
                
                ############填补完,开始分类
                
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                ####################这里batch_x改为batch_x_imp
                outputs = self.model(batch_x_imp, padding_mask, None, None)
                
                
                ###########这里加上填补损失、分类损失、总损失
                ###1)填补损失用上了上面复制的用于后续损失计算的x_imp
                x_imp = x_imp[mask==0].detach().cpu()
                batch_x = batch_x[mask==0].detach().cpu()
                ###2)分类损失
                pred = outputs.detach().cpu()
                ###3）得到填补损失、分类损失、总损失
                loss, imp_loss, ds_loss = criterion(x_imp, batch_x,pred, label.long().squeeze().cpu())
                total_loss.append(loss)
                imp_loss_total.append(imp_loss)
                ds_loss_total.append(ds_loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)
        #################
        imp_loss_total = np.average(imp_loss_total)
        ds_loss_total = np.average(ds_loss_total)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        self.imp_model.train()
        return total_loss, imp_loss_total, ds_loss_total, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            #################################
            self.imp_model.train()
            epoch_time = time.time()
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                ################
                
                #####填补开始
                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)
                
                ###输入
                batch_x_imp = self.imp_model(inp, batch_x, None, None, mask)

                ###补回去被填充的部分
                batch_x_imp = batch_x*mask + batch_x_imp*(1-mask)
                
                ###复制一个用于后续损失计算
                x_imp = batch_x_imp.clone()
                
                ############填补完,开始分类
    
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                ####和上面一样，这里batch_x改为 batch_x_imp
                outputs = self.model(batch_x_imp, padding_mask, None, None)
                #############
                loss, imp_loss, ds_loss = criterion(x_imp[mask==0], batch_x[mask==0], outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    #print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print("\titers: {0}, epoch: {1} | total_loss: {2:.7f} | imp_loss: {3:.7f} | class_loss: {4:.7f}".format(i + 1, epoch + 1, loss.item(), imp_loss.item(), ds_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            #########这里增加了vali_loss_imp, vali_loss_ds
            vali_loss, vali_loss_imp, vali_loss_ds, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, vali_loss_imp, vali_loss_ds, test_accuracy = self.vali(test_data, test_loader, criterion)
            
            ##########这里的print输出的内容略微进行调整
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}".format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            print("Epoch: {0}, Steps: {1} | Vali Imp Loss: {2:.3f} Vali CLASS Loss: {3:.3f} | Test Imp Loss: {4:.3f} Test CLASS Loss: {5:.7f}".format(epoch+1,train_steps,
                                                                                                                                                vali_loss_imp,vali_loss_ds,
                                                                                                                                                test_loss_imp,test_loss_ds))

        
            early_stopping(-val_accuracy, self.imp_model,self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        #################分别保存两个模型（填充和分类的最好参数）
        #self.model.load_state_dict(torch.load(best_model_path))
        self.imp_model.load_state_dict(torch.load(best_model_path)['imp_model'])
        self.model.load_state_dict(torch.load(best_model_path)['ds_model'])

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            #####加载在训练里面保存的两个模型的最好参数
            #self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.imp_model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))['imp_model'])
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))['ds_model'])

        #########加上imp_mse = []和imp_mae = []
        imp_mse = []
        imp_mae = []
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        ####加上self.imp_model.eval()
        self.imp_model.eval()
        self.model.eval()
        with torch.no_grad():
            ####新加两行用于后面的填补模型的mae和mse的计算
            mse_fn = nn.MSELoss()
            mae_fn = nn.L1Loss()
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                
                ########填补开始
                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)
                
                ###输入
                batch_x_imp = self.imp_model(inp, batch_x, None, None, mask)

                ###补回去被填充的部分
                batch_x_imp = batch_x*mask + batch_x_imp*(1-mask)
                
                ###复制一个用于后续损失计算
                x_imp = batch_x_imp.clone()
                
                ############填补完,开始分类
                
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                ###和上面一样，这里将batch_x改为 batch_x_imp
                outputs = self.model(batch_x_imp, padding_mask, None, None)
                
                ######新加的两行，
                batch_x = batch_x.detach().cpu()
                x_imp = x_imp.detach().cpu()

                ####新加两行代码，用于后面的填补的mse和mae的计算
                imp_mae.append(mae_fn(batch_x[mask==0],x_imp[mask==0]).item())
                imp_mse.append(mse_fn(batch_x[mask==0],x_imp[mask==0]).item())
                
                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # calculate imputaion loss
        ############这两行代码是新加的
        imp_mae = np.mean(imp_mae)
        imp_mse = np.mean(imp_mse)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        ###################这里的改动比较大，加了一句_lambda = self.args._lambda，并且print函数略改变

        print('imp_mse:{}, imp_mae:{}, accuracy:{}'.format(imp_mse, imp_mae, accuracy))
        file_name = 'result_jointly_classify.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        f.write('imp_mse:{}, imp_mae:{}, accuracy:{}'.format(imp_mse, imp_mae, accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
