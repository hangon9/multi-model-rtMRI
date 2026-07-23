
import torch
import torch.nn.functional as F

# 每个类别的样本数
samples_per_class = torch.tensor([10000, 3000, 2000, 600, 500, 300, 270], dtype=torch.float)

beta = 0.99999

effective_num = 1.0 - torch.pow(beta, samples_per_class)
weights = (1.0 - beta) / effective_num

# 归一化，让权重均值大约为 1
weights_b = weights / weights.sum() * len(samples_per_class)
weights_b = weights_b / weights_b.mean()

N = samples_per_class.sum()
C = len(samples_per_class)
print("N:", N)
print("C:", C)

weights = N / ( samples_per_class)
weights = weights / weights.mean()


print("Weights for each class CB:", weights_b)
print("Weights for each class:", weights)