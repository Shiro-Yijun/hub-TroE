import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# —————— 超参数 ———————————

# 种子
# 如果需要真随机版本，请注释掉14~15行，97~99行代码，并使用100~102行代码
# 否则请保持使用14~15行，97~99行代码，注释100~102行代码
# SEED = 42
# random.seed(SEED)

# 数据生成
# 随便写一些汉字，不包括“你”，构成一个汉字池(”你“是【目标字符】，汉字池里的是【背景字符】）
char_pool = ["我", "他", "她", "它", "们", "好", "啊", "哈", "呵", "是",
             "的", "不", "在", "有", "一", "二", "三", "四", "五", "来",
             "去", "看", "说", "听", "吃", "喝", "跑", "跳", "走", "开"]
SENTENCE_LEN = 5        # 句子长度
TOTAL_SAMPLES = 100     # 样本数量
TEST_SIZE = 0.2         # 验证集占比

# 模型
EMBEDDING_DIM = 16      # 词向量维度
HIDDEN_SIZE = 32        # RNN隐藏层维度
NUM_CLASSES = 6         # 0~5, 共6类

# 训练
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-3

# 设备
DEVICE = torch.device("cpu")
print(f"使用设备：{DEVICE}")


# —————— 1. 数据生成 ——————

# 定义数据生成函数
def generate_single_sample():
    # 第一步：随机决定是否包含“你”（1/2概率）
    has_ni = random.choice([True, False])

    if not has_ni:
        # 情况A：不包括“你”
        # 则随机选5个字
        chars  = [random.choice(char_pool) for _ in range(SENTENCE_LEN)]
        text = "".join(chars)
        label = 0
    else:
        # 情况B：包含“你”
        # 先填满5个字
        chars = [random.choice(char_pool) for _ in range(SENTENCE_LEN)]
        # 随机选一个位置（注意索引从0开始，所以位置0,1,2,3,4对应位置1,2,3,4,5）
        pos_idx = random.randint(0, SENTENCE_LEN - 1)
        # 把“你”放进去
        chars[pos_idx] = '你'
        # 拼接字符
        text = "".join(chars)
        # 标签是 位置+1（根据题目，第几位对应第几类）
        label = pos_idx + 1
    return text, label       #返回句子和标签

# 测试生成样本效果（range后面括号随便设置样本数量）
# print("测试生成的样本效果：")
# for _ in range(5):
#     print(generate_single_sample())


# —————— 数据划分 ——————  （训练集/验证集）

# 生成数据集
def generate_dataset(total_samples):
    """
    生成一个完整的数据集
    :param total_samples: 总样本数
    :return: texts列表， labels列表
    """
    texts = []
    labels = []
    for _ in range(total_samples):
        text, label = generate_single_sample()
        texts.append(text)
        labels.append(label)
    return texts, labels

# 测试生成100个样本（先写死，后续改成超参数）
all_texts, all_labels = generate_dataset(TOTAL_SAMPLES)
print(f"总样本数：{len(all_texts)}")
print(f"前5个样本：{list(zip(all_texts[:5], all_labels[:5]))}")

# 数据划分：训练集/验证集  8:2
# train_texts, val_texts, train_labels, val_labels = train_test_split(
#     all_texts, all_labels, test_size=TEST_SIZE, random_state=SEED
# )
train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_texts, all_labels, test_size=TEST_SIZE
)

print(f"训练集样本数： {len(train_texts)}")
print(f"验证集样本数： {len(val_texts)}")


# —————— 2. 预处理 ————————

# 构建词表

def build_vocab(all_texts):
    """
    从所有文本中，收集所有出现过的字，构建词表
    :param all_texts: 所有的文本列表
    :return: char2idx（字转ID）, idx2char（ID转字）
    """
    # 第一步：创建一个空集合，收集所有不重复的字
    vocab_chars = set()

    # 第二步：遍历每一句话，把每个字都加进去
    for text in all_texts:
        for char in text:
            vocab_chars.add(char)

    # 第三步：转换成有序列表
    vocab_chars = sorted(list(vocab_chars))

    # 第四步：创建映射字典
    char2idx = {char: idx for idx, char in enumerate(vocab_chars)}

    # 第五步：反向映射（后面推理展示用）
    idx2char = {idx: char for char, idx in char2idx.items()}

    return char2idx, idx2char

# 构建词表（传入所有文本）
char2idx, idx2char = build_vocab(all_texts)
VOCAB_SIZE = len(char2idx)      # 词表大小

# 测试，打印结果
# print("词表大小：", len(char2idx))
# print("字符→ID 映射：", char2idx)

# 文本转数字序列
def encode_text(text, char2idx):
    """
    将单个文本转换为ID序列
    :param text: 输入的5字文本
    :param char2idx: 字符到ID的映射
    :return: 数字序列（list）
    """
    # 遍历文本中的每个字，转换成ID
    return [char2idx[char] for char in text]

# 测试编码效果
# test_text = "你我他它她"
# test_ids = encode_text(test_text, char2idx)
# print(f"文本：{test_text}")
# print(f"编码后：{test_ids}")

# 批量编码训练集和验证集
train_ids = [encode_text(text, char2idx) for text in train_texts]
val_ids = [encode_text(text, char2idx) for text in val_texts]

# 测试打印前两个编码后的样本
# print("训练集前2个编码样本：")
# for i in range(2):
#     print(f"文本：{train_texts[i]} → 编码：{train_ids[i]} → 标签：{train_labels[i]}")


# —————— 3. 数据加载 ——————

# 自定义Dataset类

class TextDataset(Dataset):
    # 初始化,把数据传入
    def __init__(self, ids, labels):
        self.ids = ids          # 数字序列
        self.labels = labels    # 标签

    # 返回总数据量
    def __len__(self):
        return len(self.ids)

    # 返回1条数据(模型一次拿一条)
    def __getitem__(self, idx):
        # 转换成Pytorch张量
        x = torch.tensor(self.ids[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# 创建数据集
train_dataset = TextDataset(train_ids, train_labels)
val_dataset = TextDataset(val_ids, val_labels)

# 创建DataLoader

# 训练DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,       # 训练集必须打乱
    drop_last=True
)

# 验证DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 测试batch
# for x_batch, y_batch in train_loader:
#     print("x_batch 形状：", x_batch.shape)     # 应该是[4,5]，对应4句话，5个汉字
#     print("x_batch 内容：\n：", x_batch)
#     print("y_batch 形状：", y_batch.shape)     # 应该是4，对应4个标签
#     print("y_batch 内容：\n：", y_batch)

# —————— 4. 模型 ——————————

# # RNN版本
# class RNNTextClassifier(nn.Module):
#     def __init__(self):
#         super(RNNTextClassifier, self).__init__()
#
#         # 1. Embedding层：数字 → 向量
#         self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
#
#         # 2. RNN层
#         self.rnn = nn.RNN(
#             input_size=EMBEDDING_DIM,   # 输入维度 = 词向量维度
#             hidden_size=HIDDEN_SIZE,     # RNN隐藏单元数
#             batch_first=True,           # 数据形状：[batch, seq_len, dim]
#             bidirectional=False         # 单向RNN
#         )
#
#         # 3. 全连接层：输出分类
#         self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
#
#     def forward(self, x):
#         # 前向传播（数据怎么走）
#
#         # x: [batch_size, seq_len] → 如 [4,5]
#
#         # 1. 经过 Embeddig
#         out = self.embedding(x)     # out: [batch, seq_len, embedding_dim]
#
#         # 2. 经过 RNN
#         out, _ = self.rnn(out)      # out: [batch, seq_len, hidden_size]
#
#         # 3. 取最后一个时间步的输出（最关键）
#         out = out[:, -1, :]         # out: [batch, hidden_size]  (第一个“:”表示取全部样本，第二个“:”表示取全部特征，“-1”表示取最后一个位置，对应一句话里面的最后一个字）
#
#         # 4. 全连接分类
#         out = self.fc(out)
#
#         return out

# LSTM版本
class LSTMTextClassifier(nn.Module):
    def __init__(self):
        super(LSTMTextClassifier, self).__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)     #LSTM返回的第二个值是(h, c)，这里用下划线忽略
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 创建模型

# # RNN版本
# model = RNNTextClassifier().to(DEVICE)
# print(model)

# LSTM版本
model = LSTMTextClassifier().to(DEVICE)
print(model)


# —————— 5. 训练 ——————————
# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器（Adam）
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练函数

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()           # 训练模式
    total_loss = 0          # 总loss
    correct = 0             # 预测正确数量
    total = 0               # 总样本数

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # 1. 前向传播
        output = model(x)
        loss = criterion(output, y)

        # 2. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total

# 验证函数
def val_one_epoch(model, loader, criterion, device):
    model.eval()            #验证模式
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():       # 不计算梯度
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total

# 主训练循环

# 定义训练记录列表
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

print("开始训练...\n")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = val_one_epoch(model, val_loader, criterion, DEVICE)                         #这个部分也是评估环节，会体现在下面的输出内容上

    # 将数据存起来
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"[Epoch {epoch+1:2d}]"
          f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} |"
          f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

# —————— 6. 评估 ——————————

# 训练结束后，单独跑一次最终评估
final_val_loss, final_val_acc = val_one_epoch(model, val_loader, criterion, DEVICE)
print(f"\n===== 最终评估结果 =====")
print(f"验证集 Loss: {final_val_loss:.4f} | 准确率 Acc: {final_val_acc:.4f}")


# —————— 7. 推理 ——————————

def predict(text, model, char2idx, idx2char, device):     # 这里的idx2char参数只是占位用
    # 把输入文本转成模型理解的格式
    ids = [char2idx[char] for char in text]
    x = torch.tensor([ids], dtype=torch.long).to(device)

    # 模型预测
    model.eval()
    with torch.no_grad():
        output = model(x)
        pred_class = output.argmax(dim=1).item()

    # 输出结果
    if pred_class == 0:
        print(f"输入句子：{text} → 预测结果：句子中没有“你”字")
    else:
        print(f"输入句子：{text} → 预测结果：“你“字在第{pred_class}")

# 测试句子
print("\n===== 推理测试 =====")
predict("我他你它她", model, char2idx, idx2char, DEVICE)
predict("你它他她我", model, char2idx, idx2char, DEVICE)
predict("她我它他你", model, char2idx, idx2char, DEVICE)
predict("它他你我她", model, char2idx, idx2char, DEVICE)


# 训练后画折线图

# 设置图片风格
plt.rcParams['font.sans-serif'] = ['SimHei']    # 防止中文乱码
plt.rcParams['axes.unicode_minus'] = False

epochs = range(1, EPOCHS + 1)

# 画loss曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_list, 'b', label='训练集Loss')
plt.plot(epochs, val_loss_list, 'r', label='验证集Loss')
plt.title('Loss 变化曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 画Accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_list, 'b', label='训练集Loss')
plt.plot(epochs, val_acc_list, 'r', label='验证集Loss')
plt.title('Accuracy 变化曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()




