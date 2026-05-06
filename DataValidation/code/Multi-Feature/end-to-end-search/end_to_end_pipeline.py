"""
端到端聚类搜索与STIDGCN训练Pipeline

功能：
1. 从原始CSV读取数据并进行预处理
2. 提取多模块特征（DWT、相位、幅值）
3. 标准化与模块能量对齐
4. 联合搜索权重(w)和聚类数(K)
5. 筛选Top-N候选聚类方案
6. 可视化PCA聚类结果和24h平均负荷曲线
7. 对每个候选方案训练ClusterAware STIDGCN模型
8. 比较各聚类方案的预测性能

作者: zyc
日期: 2026-04-20
"""

import os
import sys
import json
import time
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 特征提取相关
import pywt
from scipy.stats import entropy
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# PyTorch相关
import torch
import torch.nn as nn
import random

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================

# 获取脚本所在目录（支持从任意位置运行）
SCRIPT_DIR = Path(__file__).parent.resolve()

# 基础路径配置（相对于脚本位置）
# 脚本位于: STLF_BDG2London/DataValidation/code/Multi-Feature/end-to-end-search/
BASE_DIR = SCRIPT_DIR.parent.parent.parent.parent.parent  # 回到STLF_BDG2London

# 默认输入文件路径（可通过命令行覆盖）
DEFAULT_INPUT_CSV = BASE_DIR / "DataValidation" / "code" / "K-shape" / "kshape_results_8784_0Start" / "sampled_8784_0Start.csv"

# STIDGCN相关路径
STIDGCN_DIR = BASE_DIR / "Cluster-aware-STIDGCN"
DATA_DIR = STIDGCN_DIR / "data" / "electricityLondon"

# 输出目录默认为脚本所在目录
DEFAULT_OUTPUT_DIR = SCRIPT_DIR

# 搜索空间配置
K_LIST = [3, 4, 5, 6, 7]
W_PHASE_LIST = [1.0, 1.5, 2.0, 2.5]
W_DWT_LIST = [0.2, 0.4, 0.6, 0.8]
W_AMP = 1.0  # 固定

# 聚类约束
MIN_CLUSTER_RATIO = 0.05  # 每个cluster至少占5%

# 训练配置
DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 64
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ============================================================
# STEP 0 & 1: 数据预处理
# ============================================================

def load_and_preprocess_data(csv_path: Path, period: int = 24) -> Tuple[np.ndarray, List[str]]:
    """
    加载CSV数据并进行缺失值填补
    
    Args:
        csv_path: CSV文件路径
        period: 周期（小时）
    
    Returns:
        data: (n_nodes, T) 的numpy数组
        node_names: 节点名称列表
    """
    logger.info(f"Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path, header=0)
    node_names = df.columns.tolist()
    n_nodes = len(node_names)
    
    # 缺失值填补
    missing = df.isna().sum().sum()
    if missing > 0:
        logger.info(f"Found {missing} missing values, using period-aligned interpolation (period={period})")
        
        for col_idx, col_name in enumerate(df.columns):
            series = df[col_name].values
            nan_idx = np.where(np.isnan(series))[0]
            
            if len(nan_idx) == 0:
                continue
            
            logger.info(f"Column `{col_name}` (col {col_idx + 1}) has {len(nan_idx)} missing values")
            
            # 第一次迭代
            for t in nan_idx:
                if t - period >= 0 and t + period < len(series):
                    if not (np.isnan(series[t - period]) or np.isnan(series[t + period])):
                        series[t] = 0.5 * (series[t - period] + series[t + period])
            
            # 多次迭代
            iteration = 1
            while np.isnan(series).any() and iteration < 10:
                remaining_before = np.isnan(series).sum()
                current_nan = np.where(np.isnan(series))[0]
                
                for t in current_nan:
                    if t - period >= 0 and t + period < len(series):
                        if not (np.isnan(series[t - period]) or np.isnan(series[t + period])):
                            series[t] = 0.5 * (series[t - period] + series[t + period])
                
                remaining_after = np.isnan(series).sum()
                iteration += 1
                
                if remaining_before == remaining_after:
                    break
            
            # 线性插值作为备选
            if np.isnan(series).any():
                remaining = np.isnan(series).sum()
                logger.info(f"  Column `{col_name}` still has {remaining} missing values, using linear interpolation")
                temp_series = pd.Series(series)
                series = temp_series.interpolate(method='linear', limit_direction='both').values
            
            df[col_name] = series
        
        logger.info("Missing value imputation completed")
    else:
        logger.info("No missing values found")
    
    # 转置为 (n_nodes, T)
    data = df.values.T
    
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Number of nodes: {n_nodes}")
    
    return data, node_names


# ============================================================
# STEP 2: 特征提取
# ============================================================

def extract_dwt_features(ts: np.ndarray, wavelet: str = "sym5", level: int = 7) -> np.ndarray:
    """
    提取DWT多尺度特征
    
    Args:
        ts: 时间序列 (T,)
        wavelet: 小波类型
        level: 分解层数
    
    Returns:
        features: (32,) 特征向量
    """
    coeffs = pywt.wavedec(ts, wavelet=wavelet, level=level)
    features = []
    
    for c in coeffs:
        c = np.asarray(c)
        mean_c = np.mean(c)
        std_c = np.std(c)
        energy_c = np.sum(c ** 2)
        
        prob = np.abs(c)
        prob = prob / (np.sum(prob) + 1e-12)
        entropy_c = entropy(prob)
        
        features.extend([mean_c, std_c, energy_c, entropy_c])
    
    return np.array(features)


def extract_phase_features(ts: np.ndarray) -> np.ndarray:
    """
    提取24h相位特征
    
    Args:
        ts: 时间序列 (T,)
    
    Returns:
        features: (6,) 特征向量
    """
    ts = ts.reshape(-1, 24)  # (days, 24)
    daily_profile = ts.mean(axis=0)  # 典型24h曲线
    
    # 峰谷检测
    peaks, _ = find_peaks(daily_profile)
    valleys, _ = find_peaks(-daily_profile)
    
    if len(peaks) > 0:
        main_peak_idx = peaks[np.argmax(daily_profile[peaks])]
        peak_hour = main_peak_idx
        peak_value = daily_profile[main_peak_idx]
        num_peaks = len(peaks)
    else:
        peak_hour = -1
        peak_value = daily_profile.max()
        num_peaks = 0
    
    if len(valleys) > 0:
        main_valley_idx = valleys[np.argmin(daily_profile[valleys])]
        valley_hour = main_valley_idx
        valley_value = daily_profile[main_valley_idx]
    else:
        valley_hour = -1
        valley_value = daily_profile.min()
    
    peak_to_valley_ratio = peak_value / (valley_value + 1e-6) if valley_value != 0 else 0
    
    return np.array([
        peak_hour, valley_hour, peak_value, valley_value,
        num_peaks, peak_to_valley_ratio
    ])


def extract_amplitude_features(ts: np.ndarray) -> np.ndarray:
    """
    提取幅值与消耗水平特征
    
    Args:
        ts: 时间序列 (T,)
    
    Returns:
        features: (7,) 特征向量
    """
    mean_load = np.mean(ts)
    std_load = np.std(ts)
    max_load = np.max(ts)
    min_load = np.min(ts)
    
    p95 = np.percentile(ts, 95)
    p5 = np.percentile(ts, 5)
    
    load_range = max_load - min_load
    cv = std_load / (mean_load + 1e-6)
    
    return np.array([
        mean_load, max_load, min_load,
        p95, p5, load_range, cv
    ])


def extract_all_features(data: np.ndarray, node_names: List[str], output_dir: Path) -> Dict[str, np.ndarray]:
    """
    提取所有特征并保存
    
    Args:
        data: (n_nodes, T) 数据
        node_names: 节点名称列表
        output_dir: 输出目录
    
    Returns:
        features_dict: 包含各模块特征的字典
    """
    n_nodes = data.shape[0]
    logger.info(f"Extracting features for {n_nodes} nodes...")
    
    # 创建特征目录
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Module 1: DWT特征 (N, 32)
    logger.info("Extracting DWT features...")
    features_dwt = np.vstack([extract_dwt_features(data[i]) for i in range(n_nodes)])
    np.save(features_dir / "dwt.npy", features_dwt)
    logger.info(f"DWT features shape: {features_dwt.shape}")
    
    # Module 2: 相位特征 (N, 6)
    logger.info("Extracting phase features...")
    features_phase = np.vstack([extract_phase_features(data[i]) for i in range(n_nodes)])
    np.save(features_dir / "phase.npy", features_phase)
    logger.info(f"Phase features shape: {features_phase.shape}")
    
    # Module 3: 幅值特征 (N, 7)
    logger.info("Extracting amplitude features...")
    features_amp = np.vstack([extract_amplitude_features(data[i]) for i in range(n_nodes)])
    np.save(features_dir / "amp.npy", features_amp)
    logger.info(f"Amplitude features shape: {features_amp.shape}")
    
    # 保存节点名称
    with open(features_dir / "node_names.json", 'w') as f:
        json.dump(node_names, f)
    
    return {
        'dwt': features_dwt,
        'phase': features_phase,
        'amp': features_amp
    }


# ============================================================
# STEP 3: 标准化与模块能量对齐
# ============================================================

def standardize_features(features_dict: Dict[str, np.ndarray], output_dir: Path) -> Dict[str, np.ndarray]:
    """
    标准化特征并进行模块能量对齐
    
    Args:
        features_dict: 包含各模块特征的字典
        output_dir: 输出目录
    
    Returns:
        standardized_features: 标准化后的特征字典
    """
    logger.info("Standardizing features...")
    
    features_std_dir = output_dir / "features_std"
    features_std_dir.mkdir(parents=True, exist_ok=True)
    
    standardized = {}
    scalers = {}
    
    for name, features in features_dict.items():
        scaler = StandardScaler()
        features_std = scaler.fit_transform(features)
        
        # 模块能量对齐：除以sqrt(dim)
        dim = features_std.shape[1]
        features_std = features_std / np.sqrt(dim)
        
        standardized[name] = features_std
        scalers[name] = scaler
        
        np.save(features_std_dir / f"{name}_std.npy", features_std)
        logger.info(f"{name} standardized shape: {features_std.shape}")
    
    # 保存scalers
    import pickle
    with open(features_std_dir / "scalers.pkl", 'wb') as f:
        pickle.dump(scalers, f)
    
    return standardized


# ============================================================
# STEP 4: 联合搜索 (w, K)
# ============================================================

def check_cluster_sizes(labels: np.ndarray, min_ratio: float = MIN_CLUSTER_RATIO) -> bool:
    """
    检查聚类大小是否满足最小比例约束
    
    Args:
        labels: 聚类标签
        min_ratio: 最小比例
    
    Returns:
        bool: 是否通过检查
    """
    n_samples = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    
    for count in counts:
        if count / n_samples < min_ratio:
            return False
    return True


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    评估聚类质量
    
    Args:
        X: 特征矩阵
        labels: 聚类标签
    
    Returns:
        metrics: 包含silhouette和DB index的字典
    """
    try:
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        return {'silhouette': sil, 'db': db}
    except Exception as e:
        logger.warning(f"Error evaluating clustering: {e}")
        return {'silhouette': -1, 'db': float('inf')}


def grid_search_clustering(
    features_std: Dict[str, np.ndarray],
    output_dir: Path,
    k_list: List[int] = K_LIST,
    w_phase_list: List[float] = W_PHASE_LIST,
    w_dwt_list: List[float] = W_DWT_LIST,
    w_amp: float = W_AMP
) -> List[Dict[str, Any]]:
    """
    网格搜索最优(w, K)组合
    
    Args:
        features_std: 标准化后的特征字典
        output_dir: 输出目录
        k_list: K值列表
        w_phase_list: w_phase权重列表
        w_dwt_list: w_dwt权重列表
        w_amp: w_amp权重（固定）
    
    Returns:
        all_results: 所有搜索结果列表
    """
    logger.info("Starting grid search for (w, K)...")
    
    clustering_dir = output_dir / "clustering_results"
    clustering_dir.mkdir(parents=True, exist_ok=True)
    
    dwt = features_std['dwt']
    phase = features_std['phase']
    amp = features_std['amp']
    
    all_results = []
    total_combinations = len(k_list) * len(w_phase_list) * len(w_dwt_list)
    current = 0
    
    for K in k_list:
        for w_phase in w_phase_list:
            for w_dwt in w_dwt_list:
                current += 1
                logger.info(f"[{current}/{total_combinations}] Testing K={K}, w_phase={w_phase}, w_dwt={w_dwt}")
                
                # 特征融合
                fused = np.hstack([
                    w_dwt * dwt,
                    w_phase * phase,
                    w_amp * amp
                ])
                
                # KMeans聚类
                kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
                labels = kmeans.fit_predict(fused)
                
                # 检查cluster size约束
                if not check_cluster_sizes(labels, MIN_CLUSTER_RATIO):
                    logger.warning(f"  Rejected: cluster size constraint violated")
                    continue
                
                # 评估指标
                metrics = evaluate_clustering(fused, labels)
                
                # 保存结果
                result_dir = clustering_dir / f"w_dwt{w_dwt}_w_phase{w_phase}_K{K}"
                result_dir.mkdir(parents=True, exist_ok=True)
                
                np.save(result_dir / "labels.npy", labels)
                
                # 计算cluster sizes
                unique, counts = np.unique(labels, return_counts=True)
                cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}
                
                result = {
                    'w_dwt': w_dwt,
                    'w_phase': w_phase,
                    'w_amp': w_amp,
                    'K': K,
                    'silhouette': float(metrics['silhouette']),
                    'db': float(metrics['db']),
                    'cluster_sizes': cluster_sizes,
                    'labels': labels.tolist()
                }
                
                with open(result_dir / "metrics.json", 'w') as f:
                    json.dump(result, f, indent=2)
                
                all_results.append(result)
                logger.info(f"  Silhouette: {metrics['silhouette']:.4f}, DB: {metrics['db']:.4f}")
    
    logger.info(f"Grid search completed. {len(all_results)} valid configurations found.")
    return all_results


# ============================================================
# STEP 5: 筛选候选并可视化
# ============================================================

def select_top_candidates(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    筛选Top-N候选聚类方案
    
    Args:
        all_results: 所有搜索结果
        output_dir: 输出目录
        top_n: 选择前N个
    
    Returns:
        selected: Top-N候选列表
    """
    logger.info(f"Selecting top {top_n} candidates...")
    
    if len(all_results) == 0:
        raise ValueError("No valid clustering results found!")
    
    # 归一化评分
    sils = np.array([r['silhouette'] for r in all_results])
    dbs = np.array([r['db'] for r in all_results])
    
    sil_norm = (sils - sils.min()) / (sils.max() - sils.min() + 1e-8)
    db_norm = (dbs - dbs.min()) / (dbs.max() - dbs.min() + 1e-8)
    
    # 综合评分: sil_norm - 0.5 * db_norm
    scores = sil_norm - 0.5 * db_norm
    
    # 排序并选择Top-N
    sorted_indices = np.argsort(scores)[::-1]
    selected = [all_results[i] for i in sorted_indices[:top_n]]
    
    # 保存结果
    selected_path = output_dir / "selected_candidates.json"
    with open(selected_path, 'w') as f:
        json.dump(selected, f, indent=2)
    
    logger.info(f"Top {top_n} candidates saved to {selected_path}")
    for i, cand in enumerate(selected):
        logger.info(f"  #{i+1}: K={cand['K']}, w_phase={cand['w_phase']}, w_dwt={cand['w_dwt']}, "
                   f"sil={cand['silhouette']:.4f}, db={cand['db']:.4f}")
    
    return selected


def visualize_candidates(
    selected: List[Dict[str, Any]],
    features_std: Dict[str, np.ndarray],
    data: np.ndarray,
    output_dir: Path
):
    """
    可视化Top-N候选聚类方案
    
    Args:
        selected: Top-N候选列表
        features_std: 标准化后的特征字典
        data: 原始数据 (n_nodes, T)
        output_dir: 输出目录
    """
    logger.info("Generating visualizations for top candidates...")
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    dwt = features_std['dwt']
    phase = features_std['phase']
    amp = features_std['amp']
    
    for idx, cand in enumerate(selected):
        logger.info(f"Visualizing candidate #{idx+1}: K={cand['K']}")
        
        cand_dir = viz_dir / f"candidate_{idx+1}_K{cand['K']}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        
        # 重构融合特征
        fused = np.hstack([
            cand['w_dwt'] * dwt,
            cand['w_phase'] * phase,
            cand['w_amp'] * amp
        ])
        
        labels = np.array(cand['labels'])
        K = cand['K']
        
        # 1. PCA可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(fused)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, K))
        
        for k in range(K):
            mask = labels == k
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[k]], label=f'Cluster {k}', s=100, alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title(f'Candidate #{idx+1}: K={K}, w_phase={cand["w_phase"]}, w_dwt={cand["w_dwt"]}\n'
                 f'Silhouette={cand["silhouette"]:.4f}, DB={cand["db"]:.4f}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(cand_dir / "pca_visualization.png", dpi=150)
        plt.close()
        
        # 2. 24h平均负荷曲线
        plt.figure(figsize=(12, 6))
        
        for k in range(K):
            mask = labels == k
            cluster_data = data[mask]  # (n_nodes_in_cluster, T)
            
            # 重塑为 (n_nodes, days, 24)
            n_days = cluster_data.shape[1] // 24
            daily_curves = cluster_data[:, :n_days*24].reshape(-1, n_days, 24)
            
            # 计算平均曲线
            avg_curve = daily_curves.mean(axis=(0, 1))  # (24,)
            
            plt.plot(range(24), avg_curve, label=f'Cluster {k} (n={mask.sum()})', 
                    linewidth=2, color=colors[k])
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Load')
        plt.title(f'Candidate #{idx+1}: 24h Average Load Curves by Cluster (K={K})')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(cand_dir / "daily_load_curves.png", dpi=150)
        plt.close()
        
        logger.info(f"  Saved visualizations to {cand_dir}")


# ============================================================
# STEP 6 & 7: 训练STIDGCN
# ============================================================

def seed_it(seed: int = 6666):
    """设置随机种子"""
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


class STIDGCNTrainer:
    """
    包装STIDGCN训练流程
    """
    
    def __init__(
        self,
        device: str,
        num_nodes: int = 68,
        channels: int = 32,
        granularity: int = 24,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 300,
        batch_size: int = 64,
        es_patience: int = 100
    ):
        self.device = torch.device(device)
        self.num_nodes = num_nodes
        self.channels = channels
        self.granularity = granularity
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.es_patience = es_patience
        
        # 导入模型 - 添加STIDGCN目录到路径
        sys.path.insert(0, str(STIDGCN_DIR))
        
        from model_ClusterAware_coarseGrained import STIDGCN, ClusterAttentionAggregation, ClusterMLPFeedback
        import util
        
        # 导入ranger21
        try:
            from ranger21 import Ranger
        except ImportError:
            # 如果ranger21未安装，使用Adam作为备选
            from torch.optim import Adam as Ranger
            logger.warning("ranger21 not found, using Adam optimizer instead")
        
        self.STIDGCN = STIDGCN
        self.ClusterAttentionAggregation = ClusterAttentionAggregation
        self.ClusterMLPFeedback = ClusterMLPFeedback
        self.Ranger = Ranger
        self.util = util
        
        # 加载数据
        self.dataloader = self.util.load_dataset(
            str(DATA_DIR), batch_size, batch_size, batch_size
        )
        self.scaler = self.dataloader["scaler"]
    
    def create_model(self, cluster_assignments: List[int]):
        """创建模型"""
        model = self.STIDGCN(
            self.device,
            input_dim=3,
            num_nodes=self.num_nodes,
            channels=self.channels,
            granularity=self.granularity,
            dropout=self.dropout
        )
        model.to(self.device)
        
        # 更新cluster_assignments
        model.cluster_assignments = cluster_assignments
        
        # 重新初始化cluster相关层 - 使用已导入的类
        model.cluster_agg = self.ClusterAttentionAggregation(
            self.channels * 2, self.num_nodes, cluster_assignments
        ).to(self.device)
        model.cluster_feedback = self.ClusterMLPFeedback(
            self.channels * 2, self.channels, self.num_nodes, cluster_assignments
        ).to(self.device)
        
        return model
    
    def train(self, cluster_assignments: List[int], save_dir: Path, log_file: Path) -> Dict[str, float]:
        """
        训练模型 - 仿照train-.py的逻辑
        
        Args:
            cluster_assignments: 聚类分配
            save_dir: 保存目录
            log_file: 日志文件路径
        
        Returns:
            metrics: 测试指标
        """
        seed_it(6666)
        
        # 创建模型
        model = self.create_model(cluster_assignments)
        optimizer = self.Ranger(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 打开日志文件
        log_f = open(log_file, 'w')
        
        def log_print(*args, **kwargs):
            """同时输出到控制台和日志文件"""
            print(*args, **kwargs)
            print(*args, file=log_f, flush=True)
        
        # 训练循环 - 仿照train-.py
        loss = 9999999.0  # 最优验证损失
        test_log = 999999.0  # 最优测试MAE
        epochs_since_best_mae = 0
        bestid = 0
        
        train_time = []
        val_time = []
        result = []
        
        log_print("start training...", flush=True)
        
        for i in range(1, self.epochs + 1):
            # 训练
            train_loss = []
            train_mape = []
            train_rmse = []
            train_wmape = []
            
            t1 = time.time()
            model.train()
            
            for iter, (x, y) in enumerate(self.dataloader["train_loader"].get_iterator()):
                trainx = torch.Tensor(x).to(self.device).transpose(1, 3)
                trainy = torch.Tensor(y).to(self.device).transpose(1, 3)
                
                optimizer.zero_grad()
                output = model(trainx).transpose(1, 3)
                real = torch.unsqueeze(trainy[:, 0, :, :], dim=1)
                predict = self.scaler.inverse_transform(output)
                
                loss_val = self.util.MAE_torch(predict, real, 0.0)
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                # 计算指标
                mape = self.util.MAPE_torch(predict, real, 0.0).item()
                rmse = self.util.RMSE_torch(predict, real, 0.0).item()
                wmape = self.util.WMAPE_torch(predict, real, 0.0).item()
                
                train_loss.append(loss_val.item())
                train_mape.append(mape)
                train_rmse.append(rmse)
                train_wmape.append(wmape)
                
                # 每50个iter打印一次
                if iter % 50 == 0:
                    log = "Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}"
                    log_print(log.format(iter, train_loss[-1], train_rmse[-1], train_mape[-1], train_wmape[-1]), flush=True)
            
            t2 = time.time()
            train_time.append(t2 - t1)
            log_print("Epoch: {:03d}, Training Time: {:.4f} secs".format(i, (t2 - t1)))
            
            # 验证
            valid_loss = []
            valid_mape = []
            valid_wmape = []
            valid_rmse = []
            
            s1 = time.time()
            model.eval()
            
            with torch.no_grad():
                for iter, (x, y) in enumerate(self.dataloader["val_loader"].get_iterator()):
                    testx = torch.Tensor(x).to(self.device).transpose(1, 3)
                    testy = torch.Tensor(y).to(self.device).transpose(1, 3)
                    
                    output = model(testx).transpose(1, 3)
                    real = torch.unsqueeze(testy[:, 0, :, :], dim=1)
                    predict = self.scaler.inverse_transform(output)
                    
                    loss_val = self.util.MAE_torch(predict, real, 0.0)
                    mape = self.util.MAPE_torch(predict, real, 0.0).item()
                    rmse = self.util.RMSE_torch(predict, real, 0.0).item()
                    wmape = self.util.WMAPE_torch(predict, real, 0.0).item()
                    
                    valid_loss.append(loss_val.item())
                    valid_mape.append(mape)
                    valid_rmse.append(rmse)
                    valid_wmape.append(wmape)
            
            s2 = time.time()
            val_time.append(s2 - s1)
            log_print("Epoch: {:03d}, Inference Time: {:.4f} secs".format(i, (s2 - s1)))
            
            # 计算平均指标
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_wmape = np.mean(train_wmape)
            mtrain_rmse = np.mean(train_rmse)
            
            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_wmape = np.mean(valid_wmape)
            mvalid_rmse = np.mean(valid_rmse)
            
            # 记录
            train_m = dict(
                train_loss=mtrain_loss,
                train_rmse=mtrain_rmse,
                train_mape=mtrain_mape,
                train_wmape=mtrain_wmape,
                valid_loss=mvalid_loss,
                valid_rmse=mvalid_rmse,
                valid_mape=mvalid_mape,
                valid_wmape=mvalid_wmape,
            )
            result.append(train_m)
            
            # 打印训练和验证指标
            log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, "
            log_print(log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape), flush=True)
            log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}"
            log_print(log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape), flush=True)
            
            # 保存训练日志
            train_csv = pd.DataFrame(result)
            train_csv.round(8).to_csv(save_dir / "train.csv", index=False)
            
            # 检查是否更新最优模型
            if mvalid_loss < loss:
                log_print("###Update tasks appear###")
                
                if i < 100:
                    # epoch < 100: 直接保存，不在test上测试
                    loss = mvalid_loss
                    torch.save(model.state_dict(), save_dir / "best_model.pth")
                    bestid = i
                    epochs_since_best_mae = 0
                    log_print("Updating! Valid Loss:", mvalid_loss, end=", ")
                    log_print("epoch: ", i)
                    
                elif i > 100:
                    # epoch > 100: 在test上测试后决定是否保存
                    outputs = []
                    realy = torch.Tensor(self.dataloader["y_test"]).to(self.device)
                    realy = realy.transpose(1, 3)[:, 0, :, :]
                    
                    with torch.no_grad():
                        for iter, (x, y) in enumerate(self.dataloader["test_loader"].get_iterator()):
                            testx = torch.Tensor(x).to(self.device).transpose(1, 3)
                            preds = model(testx).transpose(1, 3)
                            outputs.append(preds.squeeze())
                    
                    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
                    
                    # 计算12步指标
                    amae, amape, awmape, armse = [], [], [], []
                    
                    for j in range(12):
                        pred = self.scaler.inverse_transform(yhat[:, :, j])
                        real = realy[:, :, j]
                        metrics = self.util.metric(pred, real)
                        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
                        log_print(log.format(j + 1, metrics[0], metrics[2], metrics[1], metrics[3]))
                        
                        amae.append(metrics[0])
                        amape.append(metrics[1])
                        armse.append(metrics[2])
                        awmape.append(metrics[3])
                    
                    # 打印平均指标
                    log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
                    log_print(log.format(np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)))
                    
                    # 根据test MAE决定是否更新
                    if np.mean(amae) < test_log:
                        test_log = np.mean(amae)
                        loss = mvalid_loss
                        torch.save(model.state_dict(), save_dir / "best_model.pth")
                        epochs_since_best_mae = 0
                        log_print("Test low! Updating! Test Loss:", np.mean(amae), end=", ")
                        log_print("Test low! Updating! Valid Loss:", mvalid_loss, end=", ")
                        bestid = i
                        log_print("epoch: ", i)
                    else:
                        epochs_since_best_mae += 1
                        log_print("No update")
            else:
                epochs_since_best_mae += 1
                log_print("No update")
            
            # 早停检查
            if epochs_since_best_mae >= self.es_patience and i >= 300:
                log_print("Early stopping at epoch", i)
                break
        
        # 训练结束
        log_print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        log_print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        log_print("Training ends")
        log_print("The epoch of the best result：", bestid)
        
        # 加载最优模型进行最终测试
        model.load_state_dict(torch.load(save_dir / "best_model.pth"))
        model.eval()
        
        # 测试
        model.load_state_dict(torch.load(save_dir / "best_model.pth"))
        model.eval()
        
        outputs = []
        realy = torch.Tensor(self.dataloader["y_test"]).to(self.device)
        realy = realy.transpose(1, 3)[:, 0, :, :]
        
        with torch.no_grad():
            for iter, (x, y) in enumerate(self.dataloader["test_loader"].get_iterator()):
                testx = torch.Tensor(x).to(self.device).transpose(1, 3)
                preds = model(testx).transpose(1, 3)
                outputs.append(preds.squeeze())
        
        yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
        
        # 计算12步指标
        amae, amape, armse, awmape = [], [], [], []
        
        for j in range(12):
            pred = self.scaler.inverse_transform(yhat[:, :, j])
            real = realy[:, :, j]
            metrics = self.util.metric(pred, real)
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
            awmape.append(metrics[3])
        
        test_metrics = {
            'test_mae': float(np.mean(amae)),
            'test_rmse': float(np.mean(armse)),
            'test_mape': float(np.mean(amape)),
            'test_wmape': float(np.mean(awmape)),
            'best_epoch': bestid,
            'best_valid_loss': float(loss)
        }
        
        # 保存结果
        with open(save_dir / "metrics.json", 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        pd.DataFrame(result).to_csv(save_dir / "training_log.csv", index=False)
        
        logger.info(f"  Test MAE: {test_metrics['test_mae']:.4f}, "
                   f"RMSE: {test_metrics['test_rmse']:.4f}, "
                   f"MAPE: {test_metrics['test_mape']:.4f}, "
                   f"WMAPE: {test_metrics['test_wmape']:.4f}")
        
        return test_metrics


def train_all_candidates(
    selected: List[Dict[str, Any]],
    output_dir: Path,
    device: str = DEFAULT_DEVICE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> List[Dict[str, Any]]:
    """
    训练所有候选方案
    
    Args:
        selected: Top-N候选列表
        output_dir: 输出目录
        device: 计算设备
        epochs: 训练轮数
        batch_size: 批次大小
    
    Returns:
        results: 包含训练结果的字典列表
    """
    logger.info("Starting training for all candidates...")
    
    training_dir = output_dir / "training_results"
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建训练器
    trainer = STIDGCNTrainer(
        device=device,
        num_nodes=68,
        channels=32,
        granularity=24,
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.0001,
        epochs=epochs,
        batch_size=batch_size,
        es_patience=100
    )
    
    results = []
    
    for idx, cand in enumerate(selected):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training candidate #{idx+1}/{len(selected)}: K={cand['K']}")
        logger.info(f"  w_phase={cand['w_phase']}, w_dwt={cand['w_dwt']}")
        
        cand_dir = training_dir / f"candidate_{idx+1}_K{cand['K']}_wp{cand['w_phase']}_wd{cand['w_dwt']}"
        cand_log_file = cand_dir / "training.log"
        
        # 训练
        test_metrics = trainer.train(cand['labels'], cand_dir, cand_log_file)
        
        result = {
            'candidate_id': idx + 1,
            'K': cand['K'],
            'w_phase': cand['w_phase'],
            'w_dwt': cand['w_dwt'],
            'silhouette': cand['silhouette'],
            'db': cand['db'],
            **test_metrics
        }
        results.append(result)
        
        # 保存中间结果
        with open(training_dir / "results_so_far.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # 保存最终结果
    with open(training_dir / "final_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 创建对比表格
    df_results = pd.DataFrame(results)
    df_results.to_csv(training_dir / "comparison_table.csv", index=False)
    
    # 找出最佳模型
    best_idx = df_results['test_mae'].idxmin()
    best_result = results[best_idx]
    
    logger.info(f"\n{'='*60}")
    logger.info("Training completed!")
    logger.info(f"Best candidate: #{best_result['candidate_id']} (K={best_result['K']})")
    logger.info(f"  Test MAE: {best_result['test_mae']:.4f}")
    logger.info(f"  Test RMSE: {best_result['test_rmse']:.4f}")
    logger.info(f"  Test MAPE: {best_result['test_mape']:.4f}")
    logger.info(f"  Test WMAPE: {best_result['test_wmape']:.4f}")
    
    return results


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='端到端聚类搜索与STIDGCN训练Pipeline')
    parser.add_argument('--input_csv', type=str, default=str(DEFAULT_INPUT_CSV),
                       help='输入CSV文件路径')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='实验名称，将在脚本目录下创建该名称的子文件夹保存所有结果')
    parser.add_argument('--top_n', type=int, default=5,
                       help='选择Top-N个候选进行训练')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                       help='计算设备')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='批次大小')
    parser.add_argument('--skip_feature_extraction', action='store_true',
                       help='跳过特征提取（如果已存在）')
    parser.add_argument('--skip_clustering', action='store_true',
                       help='跳过聚类搜索（如果已存在）')
    parser.add_argument('--skip_training', action='store_true',
                       help='跳过训练（仅做聚类搜索）')
    
    args = parser.parse_args()
    
    # 根据实验名称创建输出目录
    output_dir = SCRIPT_DIR / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志文件
    log_file = output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("="*80)
    logger.info("End-to-End Clustering Search & STIDGCN Training Pipeline")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    
    # STEP 0 & 1: 数据预处理
    logger.info("\n" + "="*40)
    logger.info("STEP 1: Data Preprocessing")
    logger.info("="*40)
    
    data, node_names = load_and_preprocess_data(Path(args.input_csv))
    
    # 保存预处理后的数据
    np.save(output_dir / "data.npy", data)
    with open(output_dir / "node_names.json", 'w') as f:
        json.dump(node_names, f)
    
    # STEP 2: 特征提取
    features_dir = output_dir / "features"
    if args.skip_feature_extraction and (features_dir / "dwt.npy").exists():
        logger.info("\nSkipping feature extraction (using cached)...")
        features_dict = {
            'dwt': np.load(features_dir / "dwt.npy"),
            'phase': np.load(features_dir / "phase.npy"),
            'amp': np.load(features_dir / "amp.npy")
        }
    else:
        logger.info("\n" + "="*40)
        logger.info("STEP 2: Feature Extraction")
        logger.info("="*40)
        features_dict = extract_all_features(data, node_names, output_dir)
    
    # STEP 3: 标准化
    logger.info("\n" + "="*40)
    logger.info("STEP 3: Standardization")
    logger.info("="*40)
    features_std = standardize_features(features_dict, output_dir)
    
    # STEP 4: 联合搜索
    clustering_dir = output_dir / "clustering_results"
    if args.skip_clustering and clustering_dir.exists():
        logger.info("\nSkipping clustering search (using cached)...")
        # 加载已有结果
        all_results = []
        for subdir in clustering_dir.iterdir():
            if subdir.is_dir():
                with open(subdir / "metrics.json", 'r') as f:
                    all_results.append(json.load(f))
    else:
        logger.info("\n" + "="*40)
        logger.info("STEP 4: Grid Search (w, K)")
        logger.info("="*40)
        all_results = grid_search_clustering(features_std, output_dir)
    
    if len(all_results) == 0:
        logger.error("No valid clustering configurations found!")
        return
    
    # STEP 5: 筛选候选并可视化
    logger.info("\n" + "="*40)
    logger.info("STEP 5: Select Top-N Candidates & Visualization")
    logger.info("="*40)
    selected = select_top_candidates(all_results, output_dir, args.top_n)
    visualize_candidates(selected, features_std, data, output_dir)
    
    # STEP 6 & 7: 训练STIDGCN
    if not args.skip_training:
        logger.info("\n" + "="*40)
        logger.info("STEP 6 & 7: Train STIDGCN")
        logger.info("="*40)
        results = train_all_candidates(
            selected, output_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline completed successfully!")
    logger.info(f"All results saved to: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()