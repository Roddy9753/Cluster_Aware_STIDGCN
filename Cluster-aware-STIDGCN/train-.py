# 导入PyTorch库，用于深度学习模型的构建与训练
import torch
# 导入NumPy，用于数值运算
import numpy as np
# 导入Pandas，用于表格数据处理和结果保存
import pandas as pd
# 导入argparse，用于解析命令行参数
import argparse
# 导入time，用于计时
import time
# 导入自定义工具模块util，包含数据加载、度量函数等
import util
# 导入操作系统接口模块，用于文件路径及环境变量操作
import os
# 从util模块中导入所有符号
from util import *
# 导入随机模块，用于固定随机种子等
import random
# 从model模块导入STIDGCN模型类
# from model import STIDGCN
from model_ClusterAware_coarseGrained import STIDGCN
# 导入Ranger优化器（第三方优化器），用于训练
from ranger21 import Ranger
# 导入PyTorch优化器模块（在代码中被注释为备选）
import torch.optim as optim

# 以下开始使用argparse定义命令行参数
parser = argparse.ArgumentParser()
# 定义device参数，默认使用"cuda:2"，表示第3块GPU（索引从0开始）
parser.add_argument("--device", type=str, default="cuda:0", help="")
# 定义数据集名称参数，默认PEMS08
parser.add_argument("--data", type=str, default="PEMS08", help="data path")
# 定义输入维度参数，默认3
parser.add_argument("--input_dim", type=int, default=3, help="number of input_dim")
# 定义批大小参数，默认64
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
# 定义学习率参数，默认0.001
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
# 定义dropout比率，默认0.1
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
# 定义权重衰减（L2正则）参数，默认0.0001
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
# 定义训练轮数，默认500轮
parser.add_argument("--epochs", type=int, default=500, help="")
# 定义日志打印频率，默认每50个iter打印一次
parser.add_argument("--print_every", type=int, default=50, help="")
# 定义模型保存路径，默认在logs目录下并带时间戳
parser.add_argument(
    "--save",
    type=str,
    default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
    help="save path",
)
# 定义实验id，默认1
parser.add_argument("--expid", type=int, default=1, help="experiment id")
# 定义早停（early stopping）耐心值，若验证集多次无提升则提前停止，默认100
parser.add_argument(
    "--es_patience",
    type=int,
    default=100,
    help="quit if no improvement after this many iterations",
)

# 解析命令行参数并赋值到args变量
args = parser.parse_args()


# 定义训练器类trainer，封装模型、优化器、训练与评估逻辑
class trainer:
    # 类的构造函数，接收各种超参和工具对象
    def __init__(
        self,
        scaler,#归一化，反归一化工具
        input_dim,
        num_nodes,
        channels,
        dropout,
        lrate,#优化器用
        wdecay,#优化器用
        device,#算力设备
        granularity,
    ):
        # 实例化STIDGCN模型，传入设备、输入维度、节点数、通道数、时间粒度与dropout
        self.model = STIDGCN(
            device, input_dim, num_nodes, channels, granularity, dropout
        )
        # 将模型移动到指定设备（GPU或CPU）
        self.model.to(device)
        # 使用Ranger优化器对模型参数进行优化，传入学习率和权重衰减
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # 备用：可以使用Adam优化器（已注释）
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # 定义损失函数为util模块内的MAE_torch（均方误差/平均绝对误差等）
        self.loss = util.MAE_torch
        # 将传入的归一化/反归一化工具保存到实例中，用于在预测后还原数值
        self.scaler = scaler
        # 梯度裁剪阈值，防止梯度爆炸
        self.clip = 5
        # 打印模型参数量，便于了解模型规模
        print("The number of parameters: {}".format(self.model.param_num()))
        # 打印模型结构，便于调试
        print(self.model)

    # 训练步骤方法：输入input和真实值real_val，执行前向传播、计算损失并反向更新参数
    def train(self, input, real_val):
        # 切换模型到训练模式（启用dropout等训练行为）
        self.model.train()
        # 将优化器的梯度缓存清零
        self.optimizer.zero_grad()
        # 前向传播：模型输入得到输出（预测）
        output = self.model(input)
        # 对齐output与real，以便计算loss
        # 调整输出张量维度，原代码将第二和第四维交换
        output = output.transpose(1, 3)
        # 将真实值在第二维增加一个维度以匹配预测形状
        real = torch.unsqueeze(real_val, dim=1)
        # 使用scaler将模型输出从归一化空间反变换为原始值域
        predict = self.scaler.inverse_transform(output)
        # 计算损失值（此处使用MAE）
        loss = self.loss(predict, real, 0.0)
        # 反向传播计算梯度
        loss.backward()
        # 若设置了梯度裁剪，通过clip_grad_norm_裁剪梯度范数
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        # 优化器更新参数
        self.optimizer.step()
        # 计算评价指标（MAPE）并取标量值
        mape = util.MAPE_torch(predict, real, 0.0).item()
        # 计算RMSE并取标量值
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        # 计算WMAPE并取标量值
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        # 返回损失与指标（用于日志与统计）
        return loss.item(), mape, rmse, wmape

    # 评估步骤方法：不更新参数，仅进行前向计算并返回损失与指标
    def eval(self, input, real_val):
        # 切换模型到评估模式（关闭dropout等）
        self.model.eval()
        # 前向传播得到输出
        output = self.model(input)
        # 调整输出张量维度以一致
        output = output.transpose(1, 3)
        # 将真实值在第二维增加一个维度以匹配预测形状
        real = torch.unsqueeze(real_val, dim=1)
        # 将输出从归一化空间反变换到原始尺度
        predict = self.scaler.inverse_transform(output)
        # 计算损失（不反向传播）
        loss = self.loss(predict, real, 0.0)
        # 计算MAPE并取标量值
        mape = util.MAPE_torch(predict, real, 0.0).item()
        # 计算RMSE并取标量值
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        # 计算WMAPE并取标量值
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        # 返回损失与指标
        return loss.item(), mape, rmse, wmape


# 定义函数seed_it，用于设置随机种子以保证可重复性
def seed_it(seed):
    # 设置Python内置random的种子
    random.seed(seed)
    # 将种子写入环境变量PYTHONSEED
    os.environ["PYTHONSEED"] = str(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置单个CUDA设备的随机种子（用于GPU）
    torch.cuda.manual_seed(seed)
    # 设置所有可用CUDA设备的随机种子
    torch.cuda.manual_seed_all(seed)
    # 设置CuDNN为确定性模式（可能降低性能但可重复）
    torch.backends.cudnn.deterministic = True
    # 启用CuDNN（该行在设置deterministic之后仍保持True）
    torch.backends.cudnn.enabled = True
    # 设置PyTorch CPU随机种子
    torch.manual_seed(seed)


# 主函数，包含数据准备、训练循环、验证、测试与模型保存逻辑
def main():
    # 固定随机种子以获得可重复结果（这里使用6666）
    seed_it(6666)

    # 保存原始的数据集名称以便路径与命名使用
    data = args.data

    # 根据不同数据集名称设置数据路径、节点数、时间粒度与模型通道数等超参
    if args.data == "PEMS08":
        # 将数据路径指向data目录下的PEMS08
        args.data = "data//" + args.data
        # PEMS08包含170个传感器节点
        num_nodes = 170
        # 时间粒度（例如一天288个15分钟片段）
        granularity = 288
        # 模型通道数（特征维），该值会影响模型层的宽度
        channels = 48

    #'''
    #cluster aware 改变
    #'''
    elif args.data == "electricityLondon":
        # 将数据路径指向data目录下的electricityLondon
        args.data = "data//" + args.data
        # 包含68个传感器节点
        num_nodes = 68
        args.epochs = 300
        # 时间粒度（例如一天24个1h片段）
        granularity = 24
        # 模型通道数（特征维），该值会影响模型层的宽度
        channels = 32 #或48

    elif args.data == "PEMS03":
        # PEMS03数据集路径
        args.data = "data//" + args.data
        # PEMS03包含358个节点
        num_nodes = 358
        # 对于PEMS03减少训练轮数到300
        args.epochs = 300
        # 早停耐心值设为100
        args.es_patience = 100
        # 时间粒度同样为288
        granularity = 288
        # 渐进式通道配置
        channels = 32

    elif args.data == "PEMS04":
        # PEMS04数据路径
        args.data = "data//" + args.data
        # PEMS04节点数
        num_nodes = 307
        # 时间粒度
        granularity = 288
        # 通道数
        channels = 48


    elif args.data == "PEMS07":
        # PEMS07数据路径
        args.data = "data//" + args.data
        # PEMS07包含883个节点
        num_nodes = 883
        # 时间粒度
        granularity = 288
        # 通道数较大，模型更宽
        channels = 128


    elif args.data == "bike_drop":
        # 城市共享单车下车量数据集路径
        args.data = "data//" + args.data
        # 假定250个站点
        num_nodes = 250
        # 时间粒度较小（例如小时级别48）
        granularity = 48
        # 通道数
        channels = 32


    elif args.data == "bike_pick":
        # 城市共享单车上车量数据集路径
        args.data = "data//" + args.data
        # 节点数
        num_nodes = 250
        # 时间粒度
        granularity = 48
        # 通道数
        channels = 32


    elif args.data == "taxi_drop":
        # 出租车下车量数据集路径
        args.data = "data//" + args.data
        # 节点数
        num_nodes = 266
        # 时间粒度
        granularity = 48
        # 通道数更大
        channels = 96

    elif args.data == "taxi_pick":
        # 出租车上车量数据集路径
        args.data = "data//" + args.data
        # 节点数
        num_nodes = 266
        # 时间粒度
        granularity = 48
        # 通道数
        channels = 96


    # 设置PyTorch设备（GPU或CPU）
    device = torch.device(args.device)

    # 使用util.load_dataset加载数据集，返回包含train/val/test loader和scaler等的字典
    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    # 从dataloader中提取归一化/反归一化工具scaler
    scaler = dataloader["scaler"]

    # 初始化用于跟踪的最优验证损失，设为很大值
    loss = 9999999
    # 初始化用于跟踪的最优测试损失
    test_log = 999999
    # 记录自上次最佳MAE起的训练轮数，用于早停判定
    epochs_since_best_mae = 0
    # 构造模型保存路径（基于args.save与原始数据集名）
    path = args.save + data + "/"

    # 初始化历史记录列表，用于保存训练/验证时间与指标
    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []

    # 打印参数配置，便于日志追溯
    print(args)

    # 若保存路径不存在则创建（包括中间目录）
    if not os.path.exists(path):
        os.makedirs(path)

    # 实例化训练器，引入scaler、输入维度、节点数、通道、dropout、学习率、权重衰减、设备与时间粒度
    engine = trainer(
        scaler,
        args.input_dim,
        num_nodes,
        channels,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
        granularity,
    )

    # 打印训练开始标识
    print("start training...", flush=True)

    # 主训练循环：从1到给定epochs
    for i in range(1, args.epochs + 1):
        # 每个epoch初始化训练集指标列表
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        # 记录本epoch训练开始时间
        t1 = time.time()
        # 可选：dataloader['train_loader'].shuffle()（被注释）
        # 遍历训练数据迭代器，获取批量x（输入）与y（标签）
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            # 将输入x转换为PyTorch张量并移动到设备上
            trainx = torch.Tensor(x).to(device)
            # 对输入张量做维度变换以匹配模型要求（交换第2和第4维）
            trainx = trainx.transpose(1, 3)
            # 将标签y转换为张量并移动到设备
            trainy = torch.Tensor(y).to(device)
            # 对标签做同样的维度变换
            trainy = trainy.transpose(1, 3)
            # 调用engine.train执行一次训练步骤，传入trainx与trainy的第一个通道（0）
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            # 收集训练损失与指标
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

            # 每print_every个iter打印一次训练状态
            if iter % args.print_every == 0:
                log = "Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}"
                print(
                    log.format(
                        iter,
                        train_loss[-1],
                        train_rmse[-1],
                        train_mape[-1],
                        train_wmape[-1],
                    ),
                    flush=True,
                )
        # 记录本epoch训练结束时间
        t2 = time.time()
        # 打印本epoch训练耗时
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        # 将训练耗时保存到列表
        train_time.append(t2 - t1)

        # 初始化验证指标列表
        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        # 记录验证开始时间
        s1 = time.time()
        # 遍历验证集数据
        for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            # 将验证输入转换为张量并移动设备
            testx = torch.Tensor(x).to(device)
            # 调整维度
            testx = testx.transpose(1, 3)
            # 将验证标签转换为张量并移动设备
            testy = torch.Tensor(y).to(device)
            # 调整维度
            testy = testy.transpose(1, 3)
            # 使用engine.eval进行评估（不更新参数）
            metrics = engine.eval(testx, testy[:, 0, :, :])
            # 收集验证损失与指标
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])

        # 记录验证结束时间
        s2 = time.time()

        # 打印本epoch推理耗时
        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        # 保存本epoch验证耗时
        val_time.append(s2 - s1)

        # 计算本epoch训练指标均值
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)

        # 计算本epoch验证指标均值
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)

        # 将验证损失记录到历史列表
        his_loss.append(mvalid_loss)
        # 构建本epoch的指标字典，方便保存为CSV
        train_m = dict(
            train_loss=np.mean(train_loss),
            train_rmse=np.mean(train_rmse),
            train_mape=np.mean(train_mape),
            train_wmape=np.mean(train_wmape),
            valid_loss=np.mean(valid_loss),
            valid_rmse=np.mean(valid_rmse),
            valid_mape=np.mean(valid_mape),
            valid_wmape=np.mean(valid_wmape),
        )
        # 将字典转为Pandas Series以便后续拼接和保存
        train_m = pd.Series(train_m)
        # 将本epoch结果追加到结果列表
        result.append(train_m)

        # 打印本epoch训练的平均指标
        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, "
        print(
            log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape),
            flush=True,
        )
        # 打印本epoch验证的平均指标
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape),
            flush=True,
        )

        # 若本次验证损失低于历史最优，则进入更新流程
        if mvalid_loss < loss:
            print("###Update tasks appear###")
            # 若当前epoch小于100，直接保存模型为best_model.pth（模型尚未收敛，仍保存验证最优）
            if i < 100:
                # It is not necessary to print the results of the test set when epoch is less than 100, because the model has not yet converged.
                loss = mvalid_loss
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = i
                epochs_since_best_mae = 0
                print("Updating! Valid Loss:", mvalid_loss, end=", ")
                print("epoch: ", i)

            # 若当前epoch大于100，则同时在测试集上评估最佳模型并根据测试集指标决定是否更新最优模型
            elif i > 100:
                outputs = []
                # 将test集真实标签转换为张量并移动设备
                realy = torch.Tensor(dataloader["y_test"]).to(device)
                # 调整维度以匹配预测形状（转置并取第0通道）
                realy = realy.transpose(1, 3)[:, 0, :, :]

                # 遍历测试集的批次，收集模型在测试集上的预测
                for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    testx = testx.transpose(1, 3)
                    # 关闭梯度计算以节省显存和加速
                    with torch.no_grad():
                        preds = engine.model(testx).transpose(1, 3)
                    outputs.append(preds.squeeze())

                # 将所有批次预测拼接为完整的yhat
                yhat = torch.cat(outputs, dim=0)
                # 截断预测使其与真实y的样本数一致
                yhat = yhat[: realy.size(0), ...]

                # 初始化用于每个预报步长的指标列表
                amae = []
                amape = []
                awmape = []
                armse = []
                test_m = []

                # 对12个预测步长逐步计算指标（通常为12步预测）
                for j in range(12):
                    # 将第j步的预测反归一化
                    pred = scaler.inverse_transform(yhat[:, :, j])
                    # 取出真实值对应步长
                    real = realy[:, :, j]
                    # 计算各类指标，util.metric返回含多个指标的元组
                    metrics = util.metric(pred, real)
                    log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
                    print(
                        log.format(
                            j + 1, metrics[0], metrics[2], metrics[1], metrics[3]
                        )
                    )

                    # 构建并保存该步长的测试指标为Pandas Series
                    test_m = dict(
                        test_loss=np.mean(metrics[0]),
                        test_rmse=np.mean(metrics[2]),
                        test_mape=np.mean(metrics[1]),
                        test_wmape=np.mean(metrics[3]),
                    )
                    test_m = pd.Series(test_m)

                    # 将每个步长的指标列表追加到对应集合中
                    amae.append(metrics[0])
                    amape.append(metrics[1])
                    armse.append(metrics[2])
                    awmape.append(metrics[3])

                # 打印12步平均指标
                log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
                print(
                    log.format(
                        np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)
                    )
                )

                # 若12步平均MAE小于历史测试最优，则更新最优模型并保存
                if np.mean(amae) < test_log:
                    test_log = np.mean(amae)
                    loss = mvalid_loss
                    torch.save(engine.model.state_dict(), path + "best_model.pth")
                    epochs_since_best_mae = 0
                    print("Test low! Updating! Test Loss:", np.mean(amae), end=", ")
                    print("Test low! Updating! Valid Loss:", mvalid_loss, end=", ")
                    bestid = i
                    print("epoch: ", i)
                else:
                    # 若未更新，增加自上次最佳的epoch计数
                    epochs_since_best_mae += 1
                    print("No update")

        else:
            # 若验证损失无改善，增加自上次最佳的epoch计数
            epochs_since_best_mae += 1
            print("No update")

        # 将训练结果列表保存为CSV，方便外部查看训练曲线
        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{path}/train.csv")
        # 若达到早停阈值并且训练轮数不小于300，则提前终止训练
        if epochs_since_best_mae >= args.es_patience and i >= 300:
            break
    
    # 主循环结束
    # 打印平均训练时间与推理时间
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("Training ends")
    # 打印最佳模型所在epoch
    print("The epoch of the best result：", bestid)
    # 打印最佳模型对应的验证损失（保留4位小数）
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

    # 加载保存的最佳模型权重
    engine.model.load_state_dict(torch.load(path + "best_model.pth"))
    
    # 如果模型使用了ClusterGatedFeedback，启用gate记录
    if hasattr(engine.model, 'cluster_feedback') and \
       hasattr(engine.model.cluster_feedback, 'enable_gate_recording'):
        print("\n[Info] Enabling gate recording for ClusterGatedFeedback...")
        engine.model.cluster_feedback.enable_gate_recording(True)
    
    outputs = []
    # 准备测试集真实标签
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    # 在测试集上生成预测并收集
    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    # 将所有预测拼接并截断以匹配真实样本数
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    # 初始化最终测试指标列表
    amae = []
    amape = []
    armse = []
    awmape = []

    test_m = []

    # 针对12个步长计算并打印指标
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(i + 1, metrics[0], metrics[2], metrics[1], metrics[3]))

        test_m = dict(
            test_loss=np.mean(metrics[0]),
            test_rmse=np.mean(metrics[2]),
            test_mape=np.mean(metrics[1]),
            test_wmape=np.mean(metrics[3]),
        )
        test_m = pd.Series(test_m)
        test_result.append(test_m)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    # 打印12步平均测试指标
    log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)))

    # 汇总最终测试指标并保存为Pandas Series
    test_m = dict(
        test_loss=np.mean(amae),
        test_rmse=np.mean(armse),
        test_mape=np.mean(amape),
        test_wmape=np.mean(awmape),
    )
    test_m = pd.Series(test_m)
    test_result.append(test_m)

    # 将测试结果写入CSV文件以便后续分析
    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(f"{path}/test.csv")

    # 如果使用了ClusterGatedFeedback，输出gate分析报告
    if hasattr(engine.model, 'cluster_feedback') and \
       hasattr(engine.model.cluster_feedback, 'print_gate_report'):
        print("\n" + "=" * 80)
        print("[Info] Generating ClusterGatedFeedback Gate Analysis Report...")
        print("=" * 80)
        engine.model.cluster_feedback.print_gate_report()
        
        # 同时保存到文件
        import logging
        gate_log_path = f"{path}/gate_analysis.log"
        gate_logger = logging.getLogger('gate_analysis')
        gate_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(gate_log_path, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        gate_logger.addHandler(fh)
        
        engine.model.cluster_feedback.print_gate_report(gate_logger)
        gate_logger.removeHandler(fh)
        print(f"[Info] Gate analysis report saved to: {gate_log_path}")


# Python标准入口，记录总耗时并调用main
if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
