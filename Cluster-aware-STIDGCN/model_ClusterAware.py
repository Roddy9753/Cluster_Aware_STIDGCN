# =============================================================
# 导入模块
# 下面导入 PyTorch 及其常用子模块，为后续定义网络层、激活函数、张量运算等提供基础。
# =============================================================
import torch  # 导入 PyTorch 主包，用于张量运算与自动微分
import torch.nn as nn  # 导入神经网络模块别名 nn，包含各种网络层与容器
import torch.nn.functional as F  # 导入函数式接口，包含常用激活函数、损失、卷积等无状态函数
import math  # 导入数学模块，用于数学常数和函数（如开方、平方根等）


cluster_assignments = [2, 2, 0, 2, 1, 2, 1, 2, 0, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 1, 2, 1, 2, 0, 1]


# Mean aggregation
# Weighted mean
# Attention aggregation
class ClusterMeanAggregation(nn.Module):

    def __init__(self, num_nodes, cluster_assignments):
        super().__init__()

        # 把节点所属cluster的列表转换为tensor
        # 例如:
        # cluster_assignments = [0,0,1,1,2,2,...]
        # 表示 node0,node1属于cluster0
        cluster_ids = torch.tensor(cluster_assignments)

        # 计算cluster数量
        # 例如最大cluster编号=4，则cluster数=5
        self.num_clusters = int(cluster_ids.max()) + 1

        # 创建节点→cluster的矩阵
        # 维度: (num_nodes , num_clusters)
        cluster_matrix = torch.zeros(num_nodes, self.num_clusters)

        # 构造one-hot关系
        # 如果 node i 属于 cluster c
        # 那么 cluster_matrix[i,c] = 1
        for i,c in enumerate(cluster_ids):
            cluster_matrix[i,c] = 1

        # 计算每个cluster有多少节点
        # shape: (num_clusters)
        cluster_size = cluster_matrix.sum(0)

        # 为了实现 mean aggregation
        # 把矩阵每列除以 cluster_size
        # 这样每个cluster列的权重和=1
        cluster_matrix = cluster_matrix / cluster_size

        # register_buffer 表示:
        # 这是模型的一部分
        # 但不是可训练参数
        # 会自动保存到模型并自动搬到GPU
        self.register_buffer("cluster_matrix", cluster_matrix)

    def forward(self, x):

        # x 的输入维度
        # B = batch size
        # C = channel
        # N = num_nodes
        # T = time steps
        #
        # x shape = (B , C , N , T)

        cluster_matrix = self.cluster_matrix.to(x.device)

        # einsum 做矩阵乘法实现聚合
        #
        # "bcnt,nk->bckt"
        #
        # b = batch
        # c = channel
        # n = node
        # t = time
        # k = cluster
        #
        # 数学意义:
        #
        # cluster_feat[c_k] =
        #      sum_{node i in cluster k}
        #      node_feat[i] / cluster_size
        #
        # 也就是 mean aggregation
        cluster_feat = torch.einsum(
            "bcnt,nk->bckt",
            x,
            cluster_matrix
        )

        # 输出维度
        # B C K T
        #
        # K = num_clusters
        return cluster_feat


class ClusterAttentionAggregation(nn.Module):

    def __init__(self, channels, num_nodes, cluster_assignments):
        super().__init__()

        # -------------------------------------------------
        # Step 1
        # 构建 node -> cluster 关系矩阵
        # -------------------------------------------------

        # cluster_assignments:
        # 一个长度为 num_nodes 的列表
        # 例如: [0,0,0,1,1,2,2,...]
        cluster_ids = torch.tensor(cluster_assignments)

        # cluster 数量 K
        self.num_clusters = int(cluster_ids.max()) + 1

        # 创建 node-cluster 矩阵
        # 维度: (N , K)
        node_cluster = torch.zeros(num_nodes, self.num_clusters)

        for i, c in enumerate(cluster_ids):
            node_cluster[i, c] = 1

        # register_buffer:
        # 不参与训练，但会自动移动到 GPU
        self.register_buffer("node_cluster", node_cluster)


        # -------------------------------------------------
        # Step 2
        # 定义 attention scoring function φ()
        # -------------------------------------------------

        # φ(h_vi) → scalar score
        #
        # 使用一个线性层
        #
        # 输入: C
        # 输出: 1
        #
        # 即:
        #
        # score_i(t) = w^T h_vi(t)
        self.score_layer = nn.Linear(channels, 1)



    def forward(self, x):

        # x: node feature
        #
        # 维度:
        # (B , C , N , T)
        #
        # B = batch
        # C = feature channels
        # N = number of nodes
        # T = time steps


        B, C, N, T = x.shape

        node_cluster = self.node_cluster.to(x.device)


        # -------------------------------------------------
        # Step 1
        # 计算 attention score
        # -------------------------------------------------

        # 先把 tensor reshape
        #
        # (B,C,N,T) -> (B,N,T,C)
        x_perm = x.permute(0, 2, 3, 1)

        # score:
        #
        # (B,N,T,1)
        score = self.score_layer(x_perm)

        # 去掉最后一维
        #
        # (B,N,T)
        score = score.squeeze(-1)


        # -------------------------------------------------
        # Step 2
        # cluster 内 softmax
        # -------------------------------------------------

        # 创建 attention 权重 tensor
        alpha = torch.zeros_like(score)

        # 对每个 cluster 分别做 softmax
        for k in range(self.num_clusters):

            # 找到 cluster k 中的 node
            mask = node_cluster[:, k] == 1

            # 取出这些 node 的 score
            cluster_score = score[:, mask, :]

            # softmax 归一化
            cluster_alpha = torch.softmax(cluster_score, dim=1)

            # 写回 alpha
            alpha[:, mask, :] = cluster_alpha


        # -------------------------------------------------
        # Step 3
        # attention weighted aggregation
        # -------------------------------------------------

        # alpha: (B,N,T)
        # x:     (B,C,N,T)

        alpha = alpha.unsqueeze(1)

        # 加权 node feature
        weighted_node = x * alpha


        # -------------------------------------------------
        # Step 4
        # node -> cluster 聚合
        # -------------------------------------------------

        cluster_feat = torch.einsum(
            "bcnt,nk->bckt",
            weighted_node,
            node_cluster
        )

        # 输出:
        #
        # (B , C , K , T)

        return cluster_feat



# Linear feedback
# MLP feedback
# Gated feedback
class ClusterLinearFeedback(nn.Module):

    def __init__(self, channels, num_nodes, cluster_assignments):

        super().__init__()

        # 节点所属cluster
        cluster_ids = torch.tensor(cluster_assignments)

        # cluster数量
        num_clusters = int(cluster_ids.max()) + 1

        # 创建 node -> cluster 映射矩阵
        # 维度: (num_nodes , num_clusters)
        node_cluster = torch.zeros(num_nodes, num_clusters)

        # 如果 node i 属于 cluster c
        # 则 node_cluster[i,c] = 1
        for i,c in enumerate(cluster_ids):
            node_cluster[i,c] = 1

        # buffer表示固定结构，不参与训练
        self.register_buffer("node_cluster", node_cluster)

        # 定义可学习参数 W
        #
        # 作用:
        # 对cluster feature做线性变换
        #
        # 维度:
        # (channels , channels)
        self.W = nn.Parameter(torch.randn(channels, channels))

        # Xavier初始化
        nn.init.xavier_uniform_(self.W)

    def forward(self, node_feat, cluster_feat):
        # node_feat    B C N T
        # cluster_feat B C K T

        # cluster linear transform
        cluster_feat = torch.einsum(
            "cd,bckt->bdkt",
            self.W,
            cluster_feat
        )

        node_cluster = self.node_cluster.to(node_feat.device)

        # 相当于找到哪个node属于哪个cluster
        cluster_to_node = torch.einsum(
            "bdkt,nk->bdnt",
            cluster_feat,
            node_cluster
        )

        # broadcast 回 node
        return node_feat + cluster_to_node


class ClusterMLPFeedback(nn.Module):

    def __init__(self, channels, hidden_dim, num_nodes, cluster_assignments):
        super().__init__()

        # -------------------------------------------------
        # Step 1
        # 构建 node -> cluster 映射矩阵
        # -------------------------------------------------

        # 将节点所属 cluster 信息转换为 tensor
        cluster_ids = torch.tensor(cluster_assignments)

        # cluster 数量
        num_clusters = int(cluster_ids.max()) + 1

        # 创建 node-cluster 映射矩阵
        # 维度: (num_nodes , num_clusters)
        node_cluster = torch.zeros(num_nodes, num_clusters)

        # 如果 node i 属于 cluster c
        # 则 node_cluster[i,c] = 1
        for i, c in enumerate(cluster_ids):
            node_cluster[i, c] = 1

        # register_buffer 表示：
        # 这是模型结构的一部分，但不是需要训练的参数
        # 会自动保存到模型，并在 GPU / CPU 之间自动迁移
        self.register_buffer("node_cluster", node_cluster)

        # -------------------------------------------------
        # Step 2
        # 定义 MLP 参数
        # -------------------------------------------------

        # W1 : 第一层线性映射
        #
        # 作用:
        # 将 cluster feature 从 C 维映射到 hidden_dim 维
        #
        # 公式:
        #
        # z = W1 h_c
        self.W1 = nn.Parameter(torch.randn(hidden_dim, channels))

        # Xavier 初始化（深度网络常用初始化方法）
        nn.init.xavier_uniform_(self.W1)

        # W2 : 第二层线性映射
        #
        # 作用:
        # 将 hidden_dim 映射回 C 维
        #
        # 公式:
        #
        # h'_c = W2 z
        self.W2 = nn.Parameter(torch.randn(channels, hidden_dim))

        nn.init.xavier_uniform_(self.W2)

        # 非线性激活函数 ψ()
        #
        # 常用选择:
        # ReLU / GELU / Tanh
        #
        # 这里使用 ReLU
        self.activation = nn.ReLU()


    def forward(self, node_feat, cluster_feat):

        # node_feat
        # 形状: (B , C , N , T)
        #
        # B = batch size
        # C = feature channels
        # N = number of nodes
        # T = time steps

        # cluster_feat
        # 形状: (B , C , K , T)
        #
        # K = number of clusters


        # -------------------------------------------------
        # Step 1
        # 第一层线性映射
        # -------------------------------------------------

        cluster_hidden = torch.einsum(
            "hc,bckt->bhkt",
            self.W1,
            cluster_feat
        )

        # 解释：
        #
        # W1:
        # (hidden_dim , C)
        #
        # cluster_feat:
        # (B , C , K , T)
        #
        # 输出:
        #
        # cluster_hidden:
        # (B , hidden_dim , K , T)
        #
        # 数学意义:
        #
        # z = W1 * h_c


        # -------------------------------------------------
        # Step 2
        # 非线性激活函数 ψ()
        # -------------------------------------------------

        cluster_hidden = self.activation(cluster_hidden)

        # 数学意义:
        #
        # z' = ψ(z)
        #
        # 非线性激活的作用:
        #
        # 防止 MLP 退化为线性变换
        #
        # 如果没有激活函数：
        #
        # W2(W1h) = (W2W1)h
        #
        # 仍然只是线性映射


        # -------------------------------------------------
        # Step 3
        # 第二层线性映射
        # -------------------------------------------------

        cluster_out = torch.einsum(
            "ch,bhkt->bckt",
            self.W2,
            cluster_hidden
        )

        # 解释：
        #
        # W2:
        # (C , hidden_dim)
        #
        # cluster_hidden:
        # (B , hidden_dim , K , T)
        #
        # 输出:
        #
        # cluster_out:
        # (B , C , K , T)
        #
        # 数学意义:
        #
        # h'_c = W2 z'


        # -------------------------------------------------
        # Step 4
        # cluster → node broadcast
        # -------------------------------------------------

        node_cluster = self.node_cluster.to(node_feat.device)

        cluster_to_node = torch.einsum(
            "bckt,nk->bcnt",
            cluster_out,
            node_cluster
        )

        # 输入:
        #
        # cluster_out:
        # (B , C , K , T)
        #
        # node_cluster:
        # (N , K)
        #
        # 输出:
        #
        # cluster_to_node:
        # (B , C , N , T)
        #
        # 解释:
        #
        # 每个 node 接收它所属 cluster 的表示


        # -------------------------------------------------
        # Step 5
        # residual feedback
        # -------------------------------------------------

        return node_feat + cluster_to_node

        # 节点更新公式:
        #
        # h_i_new = h_i + g(h_c(i))
        #
        # 其中:
        #
        # g(h_c(i)) = W2 ψ(W1 h_c(i))


class ClusterGatedFeedback(nn.Module):

    def __init__(self, channels, num_nodes, cluster_assignments):

        super().__init__()

        # -------------------------------------------------
        # Step 1
        # 构建 node -> cluster 映射矩阵
        # -------------------------------------------------

        # cluster_assignments: 长度为 N 的列表
        # 例如 [0,0,1,1,2,2,...]
        # 表示每个 node 属于哪个 cluster
        cluster_ids = torch.tensor(cluster_assignments)

        # cluster 数量 K
        num_clusters = int(cluster_ids.max()) + 1

        # 创建 node-cluster 映射矩阵
        # 维度: (num_nodes , num_clusters)
        node_cluster = torch.zeros(num_nodes, num_clusters)

        # node i 属于 cluster c
        # 则 node_cluster[i,c] = 1
        for i, c in enumerate(cluster_ids):
            node_cluster[i, c] = 1

        # register_buffer:
        # 这个矩阵不会参与训练
        # 但会自动保存并在 CPU/GPU 间移动
        self.register_buffer("node_cluster", node_cluster)


        # -------------------------------------------------
        # Step 2
        # 定义 cluster 线性变换参数 W
        # -------------------------------------------------

        # W: (C , C)
        # 将 cluster feature 映射到 node feature 空间
        self.W = nn.Parameter(torch.randn(channels, channels))

        # Xavier 初始化
        nn.init.xavier_uniform_(self.W)


        # -------------------------------------------------
        # Step 3
        # 定义 gate 参数 Wg
        # -------------------------------------------------

        # Wg: (C , C)
        # 用于计算 gate
        #
        # γ_i(t) = sigmoid(Wg * h_vi(t))
        self.Wg = nn.Parameter(torch.randn(channels, channels))

        nn.init.xavier_uniform_(self.Wg)


        # Sigmoid 函数
        self.sigmoid = nn.Sigmoid()



    def forward(self, node_feat, cluster_feat):

        # node_feat:   (B , C , N , T)
        # cluster_feat:(B , C , K , T)


        # -------------------------------------------------
        # Step 1
        # cluster → node feature transform
        # -------------------------------------------------

        cluster_feat = torch.einsum(
            "cd,bckt->bdkt",
            self.W,
            cluster_feat
        )

        # 解释:
        #
        # W : (C , C)
        # cluster_feat : (B , C , K , T)
        #
        # 输出:
        #
        # (B , C , K , T)
        #
        # 数学意义:
        #
        # W h_ck(t)



        # -------------------------------------------------
        # Step 2
        # cluster broadcast 到 node
        # -------------------------------------------------

        node_cluster = self.node_cluster.to(node_feat.device)

        cluster_to_node = torch.einsum(
            "bdkt,nk->bdnt",
            cluster_feat,
            node_cluster
        )

        # cluster_to_node: (B , C , N , T)
        #
        # 每个 node 接收自己 cluster 的表示



        # -------------------------------------------------
        # Step 3
        # 计算 gate γ_i(t)
        # -------------------------------------------------

        gate = torch.einsum(
            "cd,bcnt->bdnt",
            self.Wg,
            node_feat
        )

        # gate shape:
        # (B , C , N , T)

        gate = self.sigmoid(gate)

        # γ_i(t) ∈ (0 , 1)
        #
        # 表示:
        #
        # 每个 node 在每个时间
        # 是否需要 cluster 信息



        # -------------------------------------------------
        # Step 4
        # gated fusion
        # -------------------------------------------------

        gated_cluster = gate * cluster_to_node

        # ⊙ 逐元素乘
        #
        # 如果 gate≈0
        #
        # 不使用 cluster 信息
        #
        # 如果 gate≈1
        #
        # 完全使用 cluster 信息



        # -------------------------------------------------
        # Step 5
        # residual update
        # -------------------------------------------------

        out = node_feat + gated_cluster

        return out


















# =============================================================
# GLU 模块（Gated Linear Unit 的变体）
# 该模块实现了带门控机制的小型 2D 卷积块：两个并行卷积产生 gate 和 value，通过 sigmoid 做门控，
# 并在随后使用 dropout 与第三个卷积进行投影。常用于通道间的非线性交互与信息过滤。
# =============================================================
class GLU(nn.Module):
    # 初始化 GLU 模块，接收特征通道数与 dropout 比例
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()  # 调用父类构造函数，初始化 nn.Module 的内部状态
        self.conv1 = nn.Conv2d(features, features, (1, 1))  # 第一个 1x1 卷积，产生 value 分支
        self.conv2 = nn.Conv2d(features, features, (1, 1))  # 第二个 1x1 卷积，产生 gate 分支
        self.conv3 = nn.Conv2d(features, features, (1, 1))  # 第三个 1x1 卷积，用于对门控后的输出做线性变换
        self.dropout = nn.Dropout(dropout)  # Dropout 层，按给定概率随机置零以防止过拟合

    # 前向传播：输入 x -> conv1, conv2 -> gate(sigmoid) * value -> dropout -> conv3 -> 返回
    def forward(self, x):
        x1 = self.conv1(x)  # 用 conv1 处理输入，得到 value 分支张量 x1
        x2 = self.conv2(x)  # 用 conv2 处理输入，得到 gate 分支张量 x2
        out = x1 * torch.sigmoid(x2)  # 对 gate 分支做 sigmoid，再与 value 分支逐元素相乘（门控机制）
        out = self.dropout(out)  # 对门控后的输出使用 dropout 随机失活
        out = self.conv3(out)  # 用 conv3 对经过 dropout 的张量做 1x1 投影
        return out  # 返回最终输出


# =============================================================
# TemporalEmbedding 类：时间/时段嵌入
# 该模块为输入的时间信息构建可训练的嵌入（按天内时间段和周内天数两部分），
# 用于将时间维度编码为与特征通道数相同的向量，以便拼接到特征图中。
# =============================================================
class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()  # 初始化父类
        self.time = time  # 存储时间粒度（例如一天内的时间段数）
        self.time_day = nn.Parameter(torch.empty(time, features))  # 可训练参数：按一天内时间段的嵌入（time x features）
        nn.init.xavier_uniform_(self.time_day)  # 使用 xavier_uniform 初始化 time_day 参数
        self.time_week = nn.Parameter(torch.empty(7, features))  # 可训练参数：按周内天数的嵌入（7 x features）
        nn.init.xavier_uniform_(self.time_week)  # 使用 xavier_uniform 初始化 time_week 参数

    # 前向传播：接收输入 x（假定最后几个维度包含时间信息），返回形状为 (batch, features, nodes, time_steps) 的张量
    def forward(self, x):
        day_emb = x[..., 1]  # 从输入 x 中取索引为 1 的时间维度（假设该维包含“日内时间编码”或相应索引）
        time_day = self.time_day[(day_emb[:, :, :] * self.time).type(torch.LongTensor)]  # 使用日内索引查表得到对应嵌入（先乘 self.time，然后取 LongTensor 作为索引）
        time_day = time_day.transpose(1, 2).contiguous()  # 交换维度并使内存连续，准备后续拼接或卷积操作

        week_emb = x[..., 2]  # 从输入 x 中取索引为 2 的时间维度（假设该维包含“周内天数编码”）
        time_week = self.time_week[(week_emb[:, :, :]).type(torch.LongTensor)]  # 使用周内索引查表得到对应嵌入
        time_week = time_week.transpose(1, 2).contiguous()  # 交换维度并使内存连续

        tem_emb = time_day + time_week  # 将日内嵌入与周内嵌入相加，产生组合时间嵌入

        tem_emb = tem_emb.permute(0,3,1,2)  # 调整维度顺序为 (batch, features, nodes, time_steps) 以便与卷积输出对齐

        return tem_emb  # 返回时间嵌入张量


# =============================================================
# Diffusion_GCN：扩散式图卷积网络（基于邻接矩阵的多步拓扑传播）
# 该模块将输入在图结构上扩散若干步（self.diffusion_step），将每一步的结果在通道维拼接后用 1x1 卷积融合，
# 并使用 dropout 防止过拟合。支持传入的邻接矩阵为 2D（全局）或 3D（每批次不同）。
# =============================================================
class Diffusion_GCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()  # 初始化父类
        self.diffusion_step = diffusion_step  # 扩散步数（在图上传播的步数）
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))  # 将拼接后的多步通道数用 1x1 卷积映射回 channels
        self.dropout = nn.Dropout(dropout)  # Dropout 层用于正则化

    # 前向传播：x 的形状通常为 (batch, channels, nodes, time)，adj 为邻接矩阵（2D 或 3D）
    def forward(self, x, adj):
        out = []  # 存储每一步扩散后的结果
        for i in range(0, self.diffusion_step):
            if adj.dim() == 3:
                # 当邻接为 3D 时（每个 batch 有不同邻接），使用 einsum 按批次矩阵相乘："bcnt,bnm->bcmt"
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x)  # 将每步结果加入列表
            elif adj.dim() == 2:
                # 当邻接为 2D 时（共享邻接），使用 einsum 进行矩阵相乘："bcnt,nm->bcmt"
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x)  # 将每步结果加入列表
        x = torch.cat(out, dim=1)  # 在通道维度将多个扩散步的输出拼接（channel 增加到 diffusion_step * channels）
        x = self.conv(x)  # 使用 1x1 卷积将拼接后的通道数映射回期望的 channels
        output = self.dropout(x)  # 应用 dropout
        return output  # 返回结果


# =============================================================
# Graph_Generator：动态图生成器
# 该模块基于输入特征与内存参数（可训练矩阵）计算节点间的相似度/动态邻接矩阵，
# 通过两种不同的相似度计算（基于内存和基于节点自身特征）产生两组邻接，融合后经过线性层、
# softmax 与 top-k 稀疏化，最终返回稀疏化的动态邻接矩阵 adj_f。
# =============================================================
class Graph_Generator(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()  # 初始化父类
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))  # 可训练内存矩阵，形状为 (channels, num_nodes)
        nn.init.xavier_uniform_(self.memory)  # 使用 xavier_uniform 初始化 memory
        self.fc = nn.Linear(2,1)  # 一个小的全连接层，用于将两路相似度信息聚合成单一标量（随后 softmax）

    # 前向传播：输入 x 形状为 (batch, channels, nodes, time)
    def forward(self, x):
        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("bcnt, cm->bnm", x, self.memory).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        # 上面：1) einsum 将 x 与 memory 做乘积得到 (batch, nodes, nodes) 类似相似度矩阵（通过 cm 投影）
        #       2) 除以 sqrt(channels) 做缩放（类似注意力缩放）
        #       3) ReLU 截断负值，再对最后一维做 softmax 归一化，得到第一种动态邻接 adj_dyn_1

        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        # 上面：计算第二种相似度：先对 time 维求和 x.sum(-1) 得到 (batch, channels, nodes)？
        # 然后通过 einsum "bcn, bcm->bnm" 得到 (batch, nodes, nodes) 的相似度矩阵，经过缩放、ReLU、softmax 得到 adj_dyn_2

        # 将两种邻接拼接到最后一维，形成 (batch, nodes, nodes, 2) 的特征以便用 fc 聚合
        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)
        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)  # 用 fc 将 2 维聚合为 1，并在最后一维做 softmax

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1]*0.8), dim=-1)  # 取 top-k（80%）最大连接
        mask = torch.zeros_like(adj_f)  # 创建与 adj_f 同形状的零掩码
        mask.scatter_(-1, topk_indices, 1)  # 在 topk 索引位置填 1，形成稀疏掩码
        adj_f = adj_f * mask  # 将 adj_f 与掩码相乘，保留 top-k，其他置零（稀疏化）

        return adj_f  # 返回稀疏化后的动态邻接矩阵


# =============================================================
# DGCN：整合图生成与扩散 GCN 的模块
# 该模块包含一个 1x1 卷积（用于投影）、一个 Graph_Generator（生成动态邻接）与 Diffusion_GCN（做图上扩散卷积）。
# 在前向中，先保存 skip（残差），再用 conv 投影，生成邻接，做 gcn，最后将 gcn 输出与嵌入相乘并加回 skip。
# =============================================================
class DGCN(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()  # 初始化父类
        self.conv = nn.Conv2d(channels,channels,(1,1))  # 1x1 卷积用于通道投影
        self.generator = Graph_Generator(channels, num_nodes, diffusion_step, dropout)  # 动态图生成器
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)  # 扩散式 GCN
        self.emb = emb  # 传入的嵌入（可能为可训练 memory），在前向用于与 gcn 输出做逐元素相乘以注入记忆

    # 前向：x -> conv 投影 -> 生成 adj_dyn -> 在 gcn 上扩散 -> 与 emb 相乘并加上 skip 残差
    def forward(self, x):
        skip = x  # 保存残差分支
        x = self.conv(x)  # 通过 1x1 卷积做通道变换
        adj_dyn = self.generator(x)  # 基于投影后的 x 生成动态邻接矩阵
        x = self.gcn(x, adj_dyn)  # 将 x 与生成的邻接传入扩散 GCN，得到图卷积后的结果
        x = x*self.emb + skip  # 将 gcn 输出与 emb 做逐元素相乘（注入记忆），再加上 skip 作为残差连接
        return x  # 返回处理后结果


# =============================================================
# Splitting：用于在时间轴上做偶数/奇数拆分的小模块
# 提供 even/odd 两个函数用于将张量按最后一维的偶数索引和奇数索引拆分，
# forward 返回一个二元组 (even, odd) 供后续交错计算使用。
# =============================================================
class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()  # 初始化父类

    def even(self, x):
        return x[:, :, :, ::2]  # 取最后一维上的偶数索引位置（步长为 2），用于交错操作

    def odd(self, x):
        return x[:, :, :, 1::2]  # 取最后一维上的奇数索引位置（从索引 1 开始，步长为 2）

    def forward(self, x):
        return (self.even(x), self.odd(x))  # 返回 (even, odd) 二元组


# =============================================================
# IDGCN：交错/可逆式图卷积块（结合了局部卷积、交错拆分、图卷积等）
# 该模块执行交错（even/odd）计算，包含四组常规卷积序列与多次 DGCN 的调用，
# 通过 x_even 与 x_odd 之间的相互作用更新时间维的偶数和奇数分量。
# =============================================================
class IDGCN(nn.Module):
    def __init__(
        self,
        device,
        channels=64,
        diffusion_step=1,
        splitting=True,
        num_nodes=170,
        dropout=0.2, emb = None
    ):
        super(IDGCN, self).__init__()  # 初始化父类

        device = device  # 接收 device（此处只是赋值，没有被用于注册设备）
        self.dropout = dropout  # 保存 dropout 比例
        self.num_nodes = num_nodes  # 节点数（图节点数量）
        self.splitting = splitting  # 是否启用拆分（若 True 则对时间维按偶/奇拆分）
        self.split = Splitting()  # 实例化 Splitting 模块用于拆分时间维

        Conv1 = []  # 用于构建第一个卷积序列的列表
        Conv2 = []  # 用于构建第二个卷积序列的列表
        Conv3 = []  # 用于构建第三个卷积序列的列表
        Conv4 = []  # 用于构建第四个卷积序列的列表
        pad_l = 3  # 左侧填充大小（用于 replication pad）
        pad_r = 3  # 右侧填充大小

        k1 = 5  # 第一个卷积核宽度
        k2 = 3  # 第二个卷积核宽度

        # 构建 Conv1 模块序列：ReplicationPad2d -> Conv2d(k1) -> LeakyReLU -> Dropout -> Conv2d(k2) -> Tanh
        Conv1 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),  # 在最后两个维度的左右两侧进行复制填充
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),  # 宽度为 k1 的 2D 卷积（1 x k1）
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # LeakyReLU 非线性激活
            nn.Dropout(self.dropout),  # Dropout 层
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),  # 另一个宽度为 k2 的卷积
            nn.Tanh(),  # Tanh 激活函数
        ]
        # Conv2 与 Conv1 结构相同（可理解为另一路并行卷积序列）
        Conv2 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        # Conv4 结构同样与 Conv1 相同（可能用于不同的变换支路）
        Conv4 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        # Conv3 结构同样与 Conv1 相同
        Conv3 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]

        # 将列表包装为 nn.Sequential，形成可调用的模块
        self.conv1 = nn.Sequential(*Conv1)  # 第一路卷积序列
        self.conv2 = nn.Sequential(*Conv2)  # 第二路卷积序列
        self.conv3 = nn.Sequential(*Conv3)  # 第三路卷积序列
        self.conv4 = nn.Sequential(*Conv4)  # 第四路卷积序列

        self.dgcn = DGCN(channels, num_nodes, diffusion_step, dropout, emb)  # 将 DGCN 实例化为成员模块，作为图卷积子模块

    # 前向传播：如果启用了拆分则按偶/奇拆分，否则以原始输入作为 even/odd；随后使用 conv 与 dgcn 进行交错更新
    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)  # 使用 split 拆分输入成偶/奇两部分
        else:
            (x_even, x_odd) = x  # 若不拆分，直接把输入当作两部分（注意：仅当外部确保 x 为二元组时有效）

        x1 = self.conv1(x_even)  # 对偶数片段使用 conv1 处理
        x1 = self.dgcn(x1)  # 经过 dgcn（图卷积）处理
        d = x_odd.mul(torch.tanh(x1))  # 用偶数片段的非线性变换影响奇数片段：x_odd * tanh(x1)

        x2 = self.conv2(x_odd)  # 对奇数片段使用 conv2 处理
        x2 = self.dgcn(x2)  # 经过 dgcn 处理
        c = x_even.mul(torch.tanh(x2))  # 用奇数片段的非线性变换影响偶数片段：x_even * tanh(x2)

        x3 = self.conv3(c)  # 对 c（由偶受奇影响的结果）使用 conv3
        x3 = self.dgcn(x3)  # 再次使用 dgcn
        x_odd_update = d + x3  # 更新奇数片段 = d（前一步） + x3（当前输出）

        x4 = self.conv4(d)  # 对 d（由奇受偶影响的结果）使用 conv4
        x4 = self.dgcn(x4)  # 使用 dgcn
        x_even_update = c + x4  # 更新偶数片段 = c（前一步） + x4（当前输出）

        return (x_even_update, x_odd_update)  # 返回更新后的偶数与奇数分量（二元组）


# =============================================================
# IDGCN_Tree：将若干 IDGCN 按树状结构组织并拼接输出
# 该模块含有三个 IDGCN 模块（使用不同的 memory 作为 emb），按特定顺序串联调用，
# 然后通过 concat 操作将其输出交错合并回原始的时间维度，并加上残差 x。
# =============================================================
class IDGCN_Tree(nn.Module):
    def __init__(
        self, device, channels=64, diffusion_step=1, num_nodes=170, dropout=0.1
    ):
        super().__init__()  # 初始化父类

        self.memory1 = nn.Parameter(torch.randn(channels, num_nodes, 6))  # 第一级记忆参数（channels x nodes x 6）
        self.memory2 = nn.Parameter(torch.randn(channels, num_nodes, 3))  # 第二级记忆参数（channels x nodes x 3）
        self.memory3 = nn.Parameter(torch.randn(channels, num_nodes, 3))  # 第三级记忆参数（channels x nodes x 3）

        # 实例化三个 IDGCN，分别传入不同的 emb（memory1, memory2, memory2）
        self.IDGCN1 = IDGCN(
            device=device,
            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout,emb=self.memory1
        )
        self.IDGCN2 = IDGCN(
            device=device,
            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout,emb=self.memory2
        )
        self.IDGCN3 = IDGCN(
            device=device,
            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout,emb=self.memory2
        )
        # 添加聚类相关模块
        self.cluster_agg1 = ClusterMeanAggregation(num_nodes, cluster_assignments)
        self.cluster_feedback1 = ClusterLinearFeedback(channels, num_nodes, cluster_assignments)
        
        self.cluster_agg2 = ClusterMeanAggregation(num_nodes, cluster_assignments)
        self.cluster_feedback2 = ClusterLinearFeedback(channels, num_nodes, cluster_assignments)
        
        self.cluster_agg3 = ClusterMeanAggregation(num_nodes, cluster_assignments)
        self.cluster_feedback3 = ClusterLinearFeedback(channels, num_nodes, cluster_assignments)

    # concat 功能：将 even 与 odd（时间维交错的两部分）按时间顺序交错合并回原始时间维
    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)  # 先变换维度以便按时间步遍历：将最后一维移到最前面
        odd = odd.permute(3, 1, 2, 0)  # 同上
        len = even.shape[0]  # 时间步数（变换后第 0 维的长度）
        _ = []  # 临时列表用于收集按时间交错的切片
        for i in range(len):
            _.append(even[i].unsqueeze(0))  # 先放入第 i 个 even 片段（加上批次维）
            _.append(odd[i].unsqueeze(0))  # 接着放入第 i 个 odd 片段（实现交错）
        return torch.cat(_, 0).permute(3, 1, 2, 0)  # 将交错后的序列连接并恢复原来的维度顺序，返回合并后的张量

    # 前向传播：先用 IDGCN1/2/3 逐步处理并得到多组输出，随后交错 concat 并与原输入相加做残差连接
    def forward(self, x):
        # 第一层处理
        cluster_feat1 = self.cluster_agg1(x)
        x_even_update1, x_odd_update1 = self.IDGCN1(x)
        x_even_update1 = self.cluster_feedback1(x_even_update1, cluster_feat1)
        x_odd_update1 = self.cluster_feedback1(x_odd_update1, cluster_feat1)
        
        # 第二层处理 - 偶数路径
        cluster_feat2 = self.cluster_agg2(x_even_update1)
        x_even_update2, x_odd_update2 = self.IDGCN2(x_even_update1)
        x_even_update2 = self.cluster_feedback2(x_even_update2, cluster_feat2)
        x_odd_update2 = self.cluster_feedback2(x_odd_update2, cluster_feat2)
        
        # 第二层处理 - 奇数路径
        cluster_feat3 = self.cluster_agg3(x_odd_update1)
        x_even_update3, x_odd_update3 = self.IDGCN3(x_odd_update1)
        x_even_update3 = self.cluster_feedback3(x_even_update3, cluster_feat3)
        x_odd_update3 = self.cluster_feedback3(x_odd_update3, cluster_feat3)
        
        # 拼接处理
        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)
        output = concat0 + x
        return output

        # x_even_update1, x_odd_update1 = self.IDGCN1(x)  # 第一次 IDGCN 处理，返回偶/奇更新
        # x_even_update2, x_odd_update2 = self.IDGCN2(x_even_update1)  # 第二次 IDGCN，作用在上一次偶更新的结果上
        # x_even_update3, x_odd_update3 = self.IDGCN3(x_odd_update1)  # 第三次 IDGCN，作用在第一次奇更新的结果上
        # concat1 = self.concat(x_even_update2, x_odd_update2)  # 将第二次的偶/奇更新交错合并
        # concat2 = self.concat(x_even_update3, x_odd_update3)  # 将第三次的偶/奇更新交错合并
        # concat0 = self.concat(concat1, concat2)  # 再次将两次合并的结果交错合并
        # output = concat0 + x  # 与原输入 x 做残差相加，作为最终输出
        # return output  # 返回输出


# =============================================================
# STIDGCN：整体时空图卷积网络（主模型）
# 该模块整合了输入投影（start_conv）、时间嵌入（Temb）、IDGCN_Tree（树状图卷积块）、
# GLU（门控单元）与回归头（regression_layer），用于从输入时空数据预测未来输出（output_len）。
# =============================================================
class STIDGCN(nn.Module):
    def __init__(
        self, device, input_dim, num_nodes, channels, granularity, dropout=0.1
    ):
        super().__init__()  # 初始化父类

        self.device = device  # 保存 device
        self.num_nodes = num_nodes  # 节点数
        self.output_len = 12  # 输出长度（预测步数），此处固定为 12（可以改为参数）
        diffusion_step = 1  # 扩散步数，固定为 1（可以改为参数）

        self.Temb = TemporalEmbedding(granularity, channels)  # 时间嵌入模块，granularity 表示日内时间粒度

        self.start_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
        )  # 输入投影层：把 input_dim 投影到 channels

        self.tree = IDGCN_Tree(
            device=device,
            channels=channels*2,
            diffusion_step=diffusion_step,
            num_nodes=self.num_nodes,
            dropout=dropout,
        )  # 树状 IDGCN 模块，注意 channels 传入为 channels*2（因为后面会拼接时间嵌入）

        self.glu = GLU(channels*2, dropout)  # GLU 门控单元，用于解码端的非线性转换

        self.regression_layer = nn.Conv2d(
            channels*2, self.output_len, kernel_size=(1, self.output_len)
        )  # 回归层：把 channels*2 的特征映射为 output_len 个时间步的预测（kernel 在时间维覆盖 output_len）

        # self.cluster_agg = ClusterMeanAggregation(num_nodes, cluster_assignments)

        # self.cluster_feedback = ClusterLinearFeedback(
        #         channels*2,
        #         num_nodes,
        #         cluster_assignments
        # )

    # 返回模型参数总数（元素数量之和）
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    # 前向传播主流程：Embedding -> Tree（IDGCN）-> Decoder (GLU + regression)
    def forward(self, input):
        x = input  # 接收输入张量（通常形状为 (batch, input_dim, nodes, time)）
        # Encoder
        # Data Embedding
        time_emb = self.Temb(input.permute(0, 3, 2, 1))  # 将输入 permute 到 TemporalEmbedding 期望的维度顺序并得到时间嵌入
        x = torch.cat([self.start_conv(x)] + [time_emb], dim=1)  # 在通道维上拼接 start_conv(x) 与 time_emb（channel 增加）

        # ---------- cluster aggregation ----------
        # cluster_feat = self.cluster_agg(x)

        # IDGCN_Tree
        x = self.tree(x)  # 将拼接后的特征送入 IDGCN_Tree 提取时空依赖

        # ---------- cluster feedback ----------
        # x = self.cluster_feedback(x, cluster_feat)

        # Decoder
        gcn = self.glu(x) + x  # 使用 GLU 处理树模块输出并做残差连接（GLU(x) + x）
        prediction = self.regression_layer(F.relu(gcn))  # 对激活后的特征用回归层预测未来 output_len 步
        return prediction  # 返回最终预测（通常形状为 (batch, output_len, nodes, 1) 或类似）
