import numpy as np
import os

# 检查PEMS08数据
print("=" * 60)
print("PEMS08 数据格式检查")
print("=" * 60)
pems08_path = "Cluster-aware-STIDGCN/data/PEMS08/train.npz"
if os.path.exists(pems08_path):
    d = np.load(pems08_path, allow_pickle=True)
    print(f"Keys: {list(d.keys())}")
    print(f"x shape: {d['x'].shape}")
    print(f"y shape: {d['y'].shape}")
    print(f"x dtype: {d['x'].dtype}")
    print(f"y dtype: {d['y'].dtype}")
    print(f"x sample [0,0,:5]: {d['x'][0,0,:5]}")
    print(f"x sample 维度说明: (samples, time_steps, nodes, features)")
else:
    print(f"文件不存在: {pems08_path}")

print()

# 检查electricityLondon数据
print("=" * 60)
print("electricityLondon 数据格式检查")
print("=" * 60)
london_path = "Cluster-aware-STIDGCN/data/electricityLondon/train.npz"
if os.path.exists(london_path):
    d = np.load(london_path, allow_pickle=True)
    print(f"Keys: {list(d.keys())}")
    print(f"x shape: {d['x'].shape}")
    print(f"y shape: {d['y'].shape}")
    print(f"x dtype: {d['x'].dtype}")
    print(f"y dtype: {d['y'].dtype}")
    print(f"x sample [0,0,:5]: {d['x'][0,0,:5]}")
    print(f"x sample 维度说明: (samples, time_steps, nodes, features)")
else:
    print(f"文件不存在: {london_path}")

print()

# 对比维度差异
print("=" * 60)
print("维度对比")
print("=" * 60)
if os.path.exists(pems08_path) and os.path.exists(london_path):
    pems08 = np.load(pems08_path, allow_pickle=True)
    london = np.load(london_path, allow_pickle=True)
    
    print(f"PEMS08 x shape:        {pems08['x'].shape}")
    print(f"electricityLondon x shape: {london['x'].shape}")
    print()
    print(f"PEMS08 y shape:        {pems08['y'].shape}")
    print(f"electricityLondon y shape: {london['y'].shape}")
    print()
    
    # 检查特征维度 (最后一个维度)
    print(f"PEMS08 features维度:        {pems08['x'].shape[-1]}")
    print(f"electricityLondon features维度: {london['x'].shape[-1]}")
    print()
    
    # 检查节点数
    print(f"PEMS08 nodes维度:        {pems08['x'].shape[2]}")
    print(f"electricityLondon nodes维度: {london['x'].shape[2]}")
    print()
    
    # 检查时间步长
    print(f"PEMS08 time_steps维度:        {pems08['x'].shape[1]}")
    print(f"electricityLondon time_steps维度: {london['x'].shape[1]}")
