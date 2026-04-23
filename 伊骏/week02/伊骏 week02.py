import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

# 使用torch计算交叉熵部分

# 先准备给交叉熵进行赋值
ce_loss = nn.CrossEntropyLoss()

# 设定样本（4个样本，5分类任务）
pred = torch.randn(4, 5)        # 生成出来的是随机数，包含了正负数和小数

# 设定符合要求的目标
target = torch.argmax(pred, dim=1)      # 用argmax函数取每分类任务中的最大值

# 计算损失函数
loss = ce_loss(pred, target)

# 打印loss函数，输出交叉熵
print(loss, "torch输出交叉熵")

# 激活函数-softmax
def softmax(logits):
    return torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)

# 验证softmax
print(torch.softmax(pred, dim=1))
print(softmax(pred))


# 手动实现部分（验算）

# 转化成onehot矩阵
def to_one_hot(target, shape=(4, 5)):
    one_hot_target = torch.zeros(shape)
    for i, j in enumerate(target):
        one_hot_target[i][j] = 1
    return one_hot_target

# 手动实现交叉熵
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred = softmax(pred)
    target = to_one_hot(target, pred.shape)
    entropy = - torch.sum(target * torch.log(pred), dim=1)
    return sum(entropy) / batch_size

print("手动实现交叉熵：",cross_entropy(pred, target))
