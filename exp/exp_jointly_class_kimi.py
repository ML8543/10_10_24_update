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
        imp_model = self.model_dict[self.args.model].Model(imp_args)
        imp_model.to(self.device)
        return imp_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # Separate optimizers for each model
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        imp_model_optim = optim.RAdam(self.imp_model.parameters(), lr=self.args.learning_rate)
        return model_optim, imp_model_optim

    def _select_criterion(self):
        imp_loss_fn = nn.MSELoss()
        ds_ls_fn = nn.CrossEntropyLoss()
        criterion = joint_loss(self.args._lambda, imp_loss_fn, ds_ls_fn)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        imp_loss_total = []
        ds_loss_total = []
        preds = []
        trues = []
        self.imp_model.eval()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                
                mask = torch.rand((batch_x.size(0), self.args.seq_len, batch_x.size(-1))).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)
                
                batch_x_imp = self.imp_model(inp, None, None, None, mask)
                batch_x_imp = batch_x*mask + batch_x_imp*(1-mask)
                
                x_imp = batch_x_imp.clone()
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                outputs = self.model(batch_x_imp, padding_mask, None, None)
                
                loss, imp_loss, ds_loss = criterion(x_imp[mask==0].detach().cpu(), 
                                                     batch_x[mask==0].detach().cpu(), 
                                                     outputs.detach().cpu(), 
                                                     label.long().squeeze().cpu())
                total_loss.append(loss)
                imp_loss_total.append(imp_loss)
                ds_loss_total.append(ds_loss)

                preds.append(outputs)
                trues.append(label)

        total_loss = np.average(total_loss)
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

        model_optim, imp_model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self.imp_model.train()
            epoch_time = time.time()
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                imp_model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                
                mask = torch.rand((batch_x.size(0), self.args.seq_len, batch_x.size(-1))).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)
                
                batch_x_imp = self.imp_model(inp, None, None, None, mask)
                batch_x_imp = batch_x*mask + batch_x_imp*(1-mask)
                
                x_imp = batch_x_imp.clone()
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                outputs = self.model(batch_x_imp, padding_mask, None, None)
                
                loss, imp_loss, ds_loss = criterion(x_imp[mask==0], 
                                                     batch_x[mask==0], 
                                                     outputs, 
                                                     label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | total_loss: {2:.7f} | imp_loss: {3:.7f} | class_loss: {4:.7f}".format(i + 1, epoch + 1, loss.item(), imp_loss.item(), ds_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
                imp_model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_loss_imp, vali_loss_ds, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test