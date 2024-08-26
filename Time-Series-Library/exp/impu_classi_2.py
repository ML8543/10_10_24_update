from data_provider.data_factory import data_provider
from exp_imputation import Exp_Imputation
from exp_classification import Exp_Classification
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, cal_accuracy
from utils.metrics import metric
from torch.utils.data import DataLoader
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
        self.device = torch.device("cuda" if self.args.use_gpu else "cpu")
        self.classification_criterion = self.classification_exp._select_criterion()
        self.train_loader = self._prepare_data_loader(args, 'train')
        self.vali_loader = self._prepare_data_loader(args, 'val')

    def _prepare_data_loader(self, args, mode):
        # 根据模式（训练或验证）获取数据集和数据加载器
        exp = Exp_Imputation(args) if mode == 'train' else Exp_Classification(args)
        data_set, data_loader = exp._get_data(flag=mode)
        return data_loader


    def train(self):
        for epoch in range(self.args.epochs):
            self._train_epoch(self.train_loader, 'train', epoch)
            self._train_epoch(self.train_loader, 'val', epoch)
            loss, accuracy = self.vali(self.vali_loader, epoch)


    def _train_epoch(self, data_loader, mode, epoch):
        model = self.imputation_model if mode == 'train' else self.classification_model
        optimizer = self.imputation_optimizer if mode == 'train' else self.classification_optimizer
        criterion = self.imputation_criterion if mode == 'train' else self.classification_criterion
        
        if mode == 'train':
            # 填补任务的训练
            model.train()
            running_loss = 0.0
            for batch_x, label, batch_x_mark, _ in data_loader:
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)

                optimizer.zero_grad()
                mask = (torch.rand_like(batch_x) < self.args.mask_rate).to(self.device).bool()
                inp = batch_x.masked_fill(mask, 0)  # 产生缺失值
                imputed_x = model(inp, batch_x_mark)  # 填补缺失值

                # 计算填补损失
                loss = criterion(imputed_x[mask], batch_x[mask])
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)

            epoch_loss = running_loss / len(data_loader.dataset)
            print(f"Epoch [{epoch+1}/{self.args.epochs}], Imputation Loss: {epoch_loss:.4f}")
        else:
            # 分类任务的训练或验证
            self.imputation_model.eval()
            
            ##这里self.classification_model.train() if mode == 'train' else     ##self.classification_model.eval()被我修改为self.classification_model.train() if mode == 'val' ##else     self.classification_model.eval()
            self.classification_model.train() if mode == 'val' else     self.classification_model.eval()
            running_loss = 0.0
            correct = 0

            with torch.no_grad():
                for batch_x, label, batch_x_mark, _ in data_loader:
                    batch_x = batch_x.to(self.device)
                    label = label.to(self.device)
                    batch_x_mark = batch_x_mark.to(self.device)

                    mask = (torch.rand_like(batch_x) < self.args.mask_rate).to(self.device).bool()
                    inp = batch_x.masked_fill(mask, 0)
                    imputed_x = self.imputation_model(inp, batch_x_mark)  # 使用填补模型

                    # 使用填补后的数据进行分类任务
                    outputs = self.classification_model(imputed_x)
                    loss = self.classification_criterion(outputs, label)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == label).sum().item()

                    running_loss += loss.item() * batch_x.size(0)

            epoch_loss = running_loss / len(data_loader.dataset)
            accuracy = 100 * correct / len(data_loader.dataset)
            print(f"Epoch [{epoch+1}/{self.args.epochs}], Classification {mode.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
    def vali(self, data_loader, epoch):
        self.imputation_model.eval()
        self.classification_model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_x, label, batch_x_mark, _ in data_loader:
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)

                # 生成随机掩码并应用到数据上
                mask = (torch.rand_like(batch_x) < self.args.mask_rate).to(self.device).bool()
                inp = batch_x.masked_fill(mask, 0)
                
                # 使用填补模型进行数据填补
                imputed_x = self.imputation_model(inp, batch_x_mark)
                
                # 计算填补损失
                imp_loss = self.imputation_criterion(imputed_x[mask], batch_x[mask])
                total_loss += imp_loss.item() * batch_x.size(0)

                # 使用填补后的数据进行分类任务
                outputs = self.classification_model(imputed_x)
                class_loss = self.classification_criterion(outputs, label)
                total_loss += class_loss.item() * batch_x.size(0)
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == label).sum().item()
                total_samples += label.size(0)

        epoch_loss = total_loss / total_samples
        accuracy = correct / total_samples
        print(f"Validation Epoch {epoch+1}: Avg Loss {epoch_loss:.4f}, Accuracy {accuracy:.4f}")

        return epoch_loss, accuracy

    def test(self, data_loader):
        self.imputation_model.eval()
        self.classification_model.eval()
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_x, label, batch_x_mark, _ in data_loader:
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)

                # 同验证过程一样，进行数据填补和分类
                mask = (torch.rand_like(batch_x) < self.args.mask_rate).to(self.device).bool()
                inp = batch_x.masked_fill(mask, 0)
                imputed_x = self.imputation_model(inp, batch_x_mark)

                outputs = self.classification_model(imputed_x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == label).sum().item()
                total_samples += label.size(0)

        accuracy = correct / total_samples
        print(f"Test Accuracy: {accuracy:.4f}")