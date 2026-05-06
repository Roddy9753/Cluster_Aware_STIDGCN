快速运行示例

nohup python -u train-.py --data electricityLondon --batch_size 64 --save "./logs/AttenAgg-GatedFb-PostAgg-K4-1" > electricityLondon-AttenAgg-GatedFb-PostAgg-K4-1.log 2>&1 &


AGG与FB组合的配置，先验聚类标签的指定，需要在model_ClusterAware_coarseGrained.py文件中手动实现


四个超过100MB的数据文件没有上传

Cluster-aware-STIDGCN/data/electricityLondon/train.npz

DataValidation/data/electricity.csv

DataValidation/electricityLondon/electricityLondon.npz

DataValidation/electricityLondon/train.npz
