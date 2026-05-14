import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------1.实现多头自注意力----------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim      # 词向量/特征维度
        self.num_heads = num_heads      # 注意力头数
        # 确保特征维度能被头数整除
        assert embed_dim % num_heads == 0, "特征维度必须能被头数整除"       # 断言，这里的用法是“assert 条件，报错信息 ”

        self.head_dim = embed_dim // num_heads      # 每个头的维度

        # 三个线性层：生成Q（查询）、K（值）、V（值）
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,query, key, value, mask=None):
        batch_size = query.shape[0]

        # 步骤1：线性变换生成Q、K、V
        Q = self.q_proj(query)      # [batch, seq_len, embed_dim]
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 步骤2，拆分成多头（batch, num_heads, seq_len, head_dim）
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 步骤3：计算注意力分数 Q*K^T / sqrt(head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # 步骤4：如果有mask,屏蔽无效位置（填充/未来未知）
        if mask is not None:
            attn_scores = attn_scores.masked_fill (mask == 0, -1e9)

        # 步骤5：softmax得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 步骤6：注意力权重 * V
        output = torch.matmul(attn_weights, V)

        # 步骤7：拼接多头
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.embed_dim)

        # 步骤8：最终线性投影
        output = self.out_proj(output)

        return output, attn_weights

# ----------2.实现transformer encoder层----------
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):      # dropwout=0.1，意思是训练时随机丢掉10%的神经元
        super().__init__()
        # 多头注意力
        self.attn = MultiHeadAttention(embed_dim, num_heads)

        # 两个层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 前馈网络FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),       # 第一层线性层，升维（embed_dim → hidden_dim）
            nn.ReLU(),                              # 激活函数，过滤无效信息，保留有效信息
            nn.Linear(hidden_dim,embed_dim)         # 第二层线性层，降维（embed_dim ← hidden_dim）
        )

        # Dropout防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 核心：残差连接 + 层归一化
        # 1. 自注意力 + 残差 +归一化
        attn_output, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前馈网络 + 残差 + 归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x

# ----------测试代码----------
if __name__ == "__main__":
    # 超参数
    embed_dim = 128     # 特征维度
    num_heads = 8       # 注意力头数
    hidden_dim = 256    # 前馈网络中间维度
    batch_size = 2      # 批次大小
    seq_len = 10        # 序列长度

    # 随机生成输入张量
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 初始化transformer层
    transformer_layer = TransformerLayer(embed_dim, num_heads, hidden_dim)

    # 前向传播
    output = transformer_layer(x)

    # 打印输入输出形状（验证维度一致）
    print("输入形状：", x.shape)
    print("输出形状：", output.shape)
