import pandas as pd

# 读取K=5聚类结果
df = pd.read_csv('DataValidation/code/Multi-Feature/Multi-Feature_result/gmm_results-1Y-Kmeans/kmeans_labels_K5.csv', index_col=0)

print('节点数量:', len(df))
print('\n聚类标签列表:')
labels = df['label'].tolist()
print(labels)

print('\n格式化为Python数组:')
print('cluster_assignments =', labels)
