# 导入自定义的数据工厂，用于根据参数提供数据
from data_provider.data_factory import data_provider
# 导入基础实验类，Exp_Main 将继承自此类
from exp.exp_basic import Exp_Basic
# 导入模型定义
from models import xPatch
# 导入工具类：早停、学习率调整、可视化
from utils.tools import EarlyStopping, adjust_learning_rate, visual
# 导入评估指标计算函数
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math  # 导入数学库，用于计算反正切等

warnings.filterwarnings('ignore')  # 忽略代码运行中的警告信息


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        # 初始化父类 Exp_Basic
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        # 模型字典，方便根据参数选择不同的模型
        model_dict = {
            'xPatch': xPatch,
        }
        # 从字典中选择并实例化模型，并将其转换为浮点类型
        model = model_dict[self.args.model].Model(self.args).float()

        # 如果使用多GPU，则使用nn.DataParallel来包装模型
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # 调用数据工厂函数获取数据集和数据加载器
        # flag可以是 'train', 'val', 'test'
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 选择优化器
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # 使用 AdamW 优化器，它在处理权重衰减时与Adam有所不同，通常在Transformer类模型上表现更好
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # # 仅使用 MSE 作为损失函数
    # def _select_criterion(self):
    #     criterion = nn.MSELoss()
    #     return criterion

    # 同时选择 MSE 和 MAE 作为损失函数
    def _select_criterion(self):
        mse_criterion = nn.MSELoss()  # 均方误差损失
        mae_criterion = nn.L1Loss()  # 平均绝对误差损失 (L1 Loss)
        return mse_criterion, mae_criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=True):
        # 验证或测试函数
        total_loss = []
        self.model.eval()  # 将模型设置为评估模式，这会关闭Dropout和BatchNorm的训练行为
        with torch.no_grad():  # 在此代码块中，禁用梯度计算以加速并节省内存
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # # batch_x 维度: (B, L, D), B=batch_size, L=seq_len, D=enc_in
                # # batch_y 维度: (B, M, D), M=label_len+pred_len
                batch_x = batch_x.float().to(self.device)  # 将输入数据转为浮点型并移动到指定设备（GPU或CPU）
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)  # 时间特征
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input (解码器输入)
                # # TODO: 此部分代码是针对 Encoder-Decoder 架构的，其中解码器需要一个起始输入。
                # # xPatch 模型没有显式的解码器，这是一个非自回归的模型。这部分代码可能是从模板中保留下来的，在xPatch中并未使用。
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                # 模型前向传播，输入 batch_x
                # # batch_x 维度: (B, L, D)
                outputs = self.model(batch_x)
                # # outputs 维度: (B, T, D), T=pred_len

                # 根据任务类型选择目标维度
                f_dim = -1 if self.args.features == 'MS' else 0
                # 截取输出的预测部分
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # 截取真实值的预测部分
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 如果在训练过程中的验证阶段 (is_test=False)，则使用带权重衰减的损失
                if not is_test:
                    # CARD loss 的权重衰减 (被注释掉)
                    # self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])

                    # Arctangent loss 的权重衰减 (对应论文公式17)
                    # 这个权重会给较近的预测点更大的权重，较远的点权重较小，但衰减速度比CARD慢
                    self.ratio = np.array([-1 * math.atan(i + 1) + math.pi / 4 + 1 for i in range(self.args.pred_len)])
                    self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')  # 增加一个维度以匹配特征维度

                    pred = outputs * self.ratio  # # (B, T, D) * (T, 1) -> (B, T, D)
                    true = batch_y * self.ratio
                else:
                    # 在最终测试时，不使用权重衰减
                    pred = outputs
                    true = batch_y

                # 计算损失
                loss = criterion(pred, true)

                total_loss.append(loss.item())  # .item() 获取纯数值
        total_loss = np.average(total_loss)  # 计算所有batch的平均损失
        self.model.train()  # 函数结束时将模型恢复到训练模式
        return total_loss

    def train(self, setting):
        # 训练主函数
        # 获取训练、验证和测试的数据加载器
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 设置模型检查点保存路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()  # 记录当前时间

        train_steps = len(train_loader)  # 一个epoch中的步数
        # 初始化早停对象
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()  # 获取优化器
        # criterion = self._select_criterion() # 如果只用MSE
        mse_criterion, mae_criterion = self._select_criterion()  # 获取MSE和MAE损失函数

        # # CARD模型的带预热的余弦学习率衰减 (被注释掉)
        # self.warmup_epochs = self.args.warmup_epochs

        # def adjust_learning_rate_new(optimizer, epoch, args):
        #     """Decay the learning rate with half-cycle cosine after warmup"""
        #     min_lr = 0
        #     if epoch < self.warmup_epochs:
        #         lr = self.args.learning_rate * epoch / self.warmup_epochs
        #     else:
        #         lr = min_lr+ (self.args.learning_rate - min_lr) * 0.5 * \
        #             (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.args.train_epochs - self.warmup_epochs)))

        #     for param_group in optimizer.param_groups:
        #         if "lr_scale" in param_group:
        #             param_group["lr"] = lr * param_group["lr_scale"]
        #         else:
        #             param_group["lr"] = lr
        #     print(f'Updating learning rate to {lr:.7f}')
        #     return lr

        # # train_times = [] # 用于计算成本分析

        # 开始训练循环
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            # # train_time = 0 # 用于计算成本分析

            self.model.train()  # 将模型设置为训练模式
            epoch_time = time.time()  # 记录一个epoch开始的时间
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()  # 清空上一轮的梯度
                # # batch_x 维度: (B, L, D)
                batch_x = batch_x.float().to(self.device)

                # # batch_y 维度: (B, M, D)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # # TODO: 同vali函数，这部分代码在xPatch中未使用
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 模型前向传播
                # # temp = time.time() # 用于计算成本分析
                outputs = self.model(batch_x)
                # # train_time += time.time() - temp # 用于计算成本分析

                # 截取预测部分
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # CARD loss 的权重衰减 (被注释掉)
                # self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])

                # Arctangent loss 的权重衰减 (对应论文公式17)
                self.ratio = np.array([-1 * math.atan(i + 1) + math.pi / 4 + 1 for i in range(self.args.pred_len)])
                self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')

                # 将权重应用到输出和真实值上
                outputs = outputs * self.ratio
                batch_y = batch_y * self.ratio

                # 使用加权后的MAE作为训练损失 (对应论文公式15)
                loss = mae_criterion(outputs, batch_y)

                # loss = criterion(outputs, batch_y) # 如果使用MSE

                train_loss.append(loss.item())

                # 每100个iter打印一次训练信息
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()  # 反向传播计算梯度
                model_optim.step()  # 优化器更新权重

            # # train_times.append(train_time/len(train_loader)) # 用于计算成本分析
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            # 在每个epoch结束后，在验证集和测试集上进行评估
            # vali_loss = self.vali(vali_data, vali_loader, criterion) # 如果只用MSE
            # test_loss = self.vali(test_data, test_loader, criterion) # 如果只用MSE

            # 使用加权MAE计算验证集损失
            vali_loss = self.vali(vali_data, vali_loader, mae_criterion, is_test=False)
            # 使用标准的MSE计算测试集损失
            test_loss = self.vali(test_data, test_loader, mse_criterion, is_test=True)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # 调用早停，传入验证集损失。如果损失不再下降，会保存模型并可能提前停止训练
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            # adjust_learning_rate_new(model_optim, epoch + 1, self.args)

            # # print('Alpha:', self.model.decomp.ma.alpha) # 打印学习到的alpha
            # # print('Beta:', self.model.decomp.ma.beta)   # 打印学习到的beta

        # # print("Training time: {}".format(np.sum(train_times)/len(train_times))) # 用于计算成本分析

        # 加载早停法保存的最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # # TODO: 这里加载完最佳模型后，直接删除了模型文件。这假设模型状态已在 self.model 中，并且当前运行不再需要该文件。
        os.remove(best_model_path)

        return self.model

    def test(self, setting, test=0):
        # 测试函数
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            # 加载已经训练好的模型
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []  # 存储预测结果
        trues = []  # 存储真实结果
        folder_path = './test_results/' + setting + '/'  # 可视化结果保存路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # # test_time = 0 # 用于计算成本分析
        self.model.eval()  # 模型设为评估模式
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # # TODO: 同上，dec_inp 在 xPatch 中未使用
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 模型前向传播
                # # temp = time.time() # 用于计算成本分析
                outputs = self.model(batch_x)
                # # test_time += time.time() - temp # 用于计算成本分析

                # 截取预测部分
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 将tensor转为numpy数组
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                # 每20个batch保存一次可视化结果
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    # 将输入序列的最后一个维度（目标变量）和真实值、预测值拼接起来，用于画图
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # # print("Inference time: {}".format(test_time/len(test_loader))) # 用于计算成本分析
        preds = np.array(preds)  # # 形状: (num_batches, B, T, D)
        trues = np.array(trues)
        # # 下面被注释掉的concatenate适用于dataloader不使用drop_last=True的情况
        # preds = np.concatenate(preds, axis=0)
        # trues = np.concatenate(trues, axis=0)

        # 将批次和批内样本两个维度合并，得到 (总样本数, T, D)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # 计算评估指标
        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # 将结果写入文件
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # # np.save(...) # 保存更详细的结果
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return
