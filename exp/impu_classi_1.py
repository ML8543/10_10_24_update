from data_provider.data_factory import data_provider

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, cal_accuracy
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb


warnings.filterwarnings('ignore')

# 定义联合训练的类
class Exp_JointTraining:
    def __init__(self, args):
        self.args = args
        self.imputation_exp = Exp_Imputation(args)
        self.classification_exp = Exp_Classification(args)
        self.imputation_model = self.imputation_exp._build_model()
        self.classification_model = self.classification_exp._build_model()
        self.model = nn.ModuleList([self.imputation_model, self.classification_model])
        self.imputation_optimizer = self.imputation_exp._select_optimizer()
        self.classification_optimizer = self.classification_exp._select_optimizer()
        self.imputation_criterion = self.imputation_exp._select_criterion()
        self.classification_criterion = self.classification_exp._select_criterion()
        self.train_loader = self._prepare_data_loader(args, 'train')
        self.vali_loader = self._prepare_data_loader(args, 'val')

    def _prepare_data_loader(self, args, mode):
        # 根据模式（训练或验证）获取数据集和数据加载器
        exp = Exp_Imputation(args) if mode == 'train' else Exp_Classification(args)
        data_set, data_loader = exp._get_data(flag=mode)
        return data_loader


    def train(self):
        # 获取数据
        train_data, train_loader = self.imputation_exp._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self.imputation_exp._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        time_now = time.time()

        train_steps = len(train_loader)
        # 初始化EarlyStopping
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 联合训练循环
        for epoch in range(self.args.train_epochs):
            self._train_epoch(train_loader, epoch, early_stopping, path)
            
        
        ##联合训练的一个循环的代码    
        def _train_epoch(self, train_loader, epoch, early_stopping, path):
            train_loss = []
            iter_count = 0
            
            self.model[0].train()  # 填补模型训练模式
            epoch_time = time.time()

            for i, (batch_x, _, batch_x_mark, _) in enumerate(train_loader):
                # 填补数据
                batch_x = batch_x.float().to(self.args.device)
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                mask = torch.rand_like(batch_x) < self.args.mask_rate
                inp = batch_x.masked_fill(mask, 0)
                imputed_x = self.imputation_model(inp, batch_x_mark)

                # 计算填补损失
                imputation_loss = self.imputation_criterion(imputed_x[mask], batch_x[mask])
                train_loss.append(imputation_loss.item())
                
                # 更新填补模型
                self.imputation_optimizer.zero_grad()
                imputation_loss.backward()
                self.imputation_optimizer.step()
                
                self.model[1].train()  # 分类模型训练模式
                classification_outputs = self.classification_model(imputed_x, None, None)
                classification_loss = self.classification_criterion(classification_outputs,     batch_x_mark)
                total_loss = imputation_loss + classification_loss

                # 更新分类模型
                self.classification_optimizer.zero_grad()
                total_loss.backward()
                self.classification_optimizer.step()

                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.args.train_epochs}], Step     [{i+1}/{len(train_loader)}], "
                      f"Imputation Loss: {imputation_loss.item():.4f}, "
                      f"Classification Loss: {classification_loss.item():.4f}")
