from utils.data_preprocessing import Big_data
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from collections import Counter
train_dataset = Big_data('../data/BenchmarkDatasets', dataset_type='train', horizon=5, observation_length=10,
                         train_val_split=0.8, n_trends=3)

dataset = DataLoader(train_dataset,batch_size=32,
                            shuffle=False)

sample = next(iter(dataset))
features, label = sample
# all_labels = []
# for _, label in train_dataset:
#     all_labels.extend(label.flatten().tolist())
# label_counts = torch.bincount(torch.tensor(all_labels))
# print(label_counts)
all_labels = []
for _, label in train_dataset:
    all_labels.extend(label.flatten().tolist())

label_counts = torch.bincount(torch.tensor(all_labels))
# label_counts = Counter(all_labels)
#
# print(f"Label 0: {label_counts[0]} instances")
# print(f"Label 1: {label_counts[1]} instances")
# print(f"Label 2: {label_counts[2]} instances")
# for i, count in enumerate(label_counts):
#     print(f"Label {i}: {count} instances")
plt.pie(label_counts, labels=['D','S','U'], colors = ['seagreen', 'darkgray', 'darkturquoise'])
plt.show()