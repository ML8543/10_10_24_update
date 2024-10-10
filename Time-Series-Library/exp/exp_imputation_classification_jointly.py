import sys
sys.path.append('/root/autodl-tmp/Time-Series-Library/exp')
from data_provider.data_factory import data_provider
from exp.exp_imputation import Exp_Imputation
from exp.exp_classification import Exp_Classification
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
class Exp_Imputation_Classification_Jointly_LossImp:
    def __init__(self, args):
        self.args = args
        ##imputation_classification_jointly_imploss
        args.task_name = 'imputation'
        self.imputation_exp = Exp_Imputation(args)

        args.task_name = 'classification'
        # args.data = 'UEA'
        self.classification_exp = Exp_Classification(args)

        args.task_name = 'imputation'
        self.imputation_model = self.imputation_exp._build_model()
        args.task_name = 'classification'
        self.classification_model = self.classification_exp._build_model()
        #self.model = nn.ModuleList([self.imputation_model, self.classification_model])
        self.imputation_optimizer = self.imputation_exp._select_optimizer()
        self.classification_optimizer = self.classification_exp._select_optimizer()
        self.imputation_criterion = self.imputation_exp._select_criterion()
        self.device = torch.device("cuda" if self.args.use_gpu else "cpu")
        self.classification_criterion = self.classification_exp._select_criterion()
        args.task_name = 'imputation'
        self.train_loader = self._prepare_data_loader(args, 'train')
        self.vali_loader = self._prepare_data_loader(args, 'val')
        self.best_vali_loss = float('inf')  # 最佳验证损失初始化为无穷大
        self.best_model_path = os.path.join(self.args.checkpoints, 'best_model.pth')  # 最佳模型保存路径

    def _prepare_data_loader(self, args, mode):
        # 根据模式（训练或验证）获取数据集和数据加载器
        exp = Exp_Imputation(args) if mode == 'train' else Exp_Classification(args)
        data_set, data_loader = exp._get_data(flag=mode)
        return data_loader


    def train(self):
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, path=self.best_model_path)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        for epoch in range(self.args.epochs):
            self._train_epoch(self.train_loader, epoch)
            # 在这里调用验证方法来获取验证损失
            vali_loss, accuracy = self.vali(self.vali_loader, epoch)
            
            # 更新 EarlyStopping 状态
            self.early_stopping(vali_loss, self.model, epoch)
            
            # 如果 EarlyStopping 决定停止，则中断循环
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            
            # 如果当前模型的验证损失最低，则保存模型
            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved at epoch {epoch+1}")
        # 加载最佳模型的状态
        self.model.load_state_dict(torch.load(self.best_model_path))
        
        # 返回训练后的模型
        return self.model

        

    def _train_epoch(self, data_loader, mode, epoch):
        # 同时训练填补模型和分类模型
        self.imputation_model.train()
        self.classification_model.train()
    
        running_loss = 0.0
        for batch_x, label, batch_x_mark, _ in data_loader:
            batch_x = batch_x.to(self.device)
            label = label.to(self.device)
            batch_x_mark = batch_x_mark.to(self.device)

            # 清空梯度
            self.imputation_optimizer.zero_grad()
            self.classification_optimizer.zero_grad()  # 确保分类模型的梯度也被清空

            mask = (torch.rand_like(batch_x) < self.args.mask_rate).to(self.device).bool()
            inp = batch_x.masked_fill(mask, 0)  # 产生缺失值
            imputed_x = self.imputation_model(inp, batch_x_mark)  # 填补缺失值

            # 计算填补损失
            imputation_loss = self.imputation_criterion(imputed_x[mask], batch_x[mask])

            # 使用填补后的数据进行分类任务的前向传播
            classification_outputs = self.classification_model(imputed_x)
        
            # 假设分类任务使用CrossEntropyLoss，且label已经是one-hot格式或者已经调整为适配模型输出的形式
            # 如果label不是one-hot格式，您需要根据实际情况调整这里的计算方式
            classification_loss = self.classification_criterion(classification_outputs, label)

            # 将填补损失和分类损失合并作为总损失
            total_loss = imputation_loss + 0*classification_loss


            # 反向传播和参数更新
            total_loss.backward()
            self.imputation_optimizer.step()  # 更新填补模型参数
            self.classification_optimizer.step()  # 更新分类模型参数

            running_loss += total_loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)
        print(f"Epoch [{epoch+1}/{self.args.epochs}], Total Loss: {epoch_loss:.4f}")
        
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

        vali_loss = total_loss / total_samples
        accuracy = correct / total_samples
        print(f"Validation Epoch {epoch+1}: Avg Loss {vali_loss:.4f}, Accuracy {accuracy:.4f}")

        return vali_loss, accuracy

    def test(self, data_loader):
        # 加载保存的最佳模型状态
        self.model.load_state_dict(torch.load(self.best_model_path))
    
        # 确保模型处于评估模式
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
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        result_filename = 'result_classification.txt'
        with open(os.path.join(folder_path, result_filename), 'a') as f:
            f.write(setting + "  \n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write('\n')
            f.write('\n')
            f.close()
            return
