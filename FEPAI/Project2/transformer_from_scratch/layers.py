# transformer_from_scratch/layers.py

import torch
import torch.nn as nn
import math


class PositionwiseFeedForward(nn.Module):
    """
    原生实现位置前馈网络 (Position-wise Feed-Forward Network)。
    由两个线性层和一个ReLU激活函数组成。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): 模型的输入输出维度。
            d_ff (int): 中间隐藏层的维度，通常是 d_model 的 4 倍。
            dropout (float, optional): Dropout概率. Defaults to 0.1.
        """
        super().__init__()
        # --- YOUR CODE HERE ---
        # 定义两个线性层和激活函数
        # 第一个线性层: d_model -> d_ff
        # 第二个线性层: d_ff -> d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        # --- END YOUR CODE ---

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量, shape [B, L, d_model]

        Returns:
            torch.Tensor: 输出张量, shape [B, L, d_model]
        """
        residual = x

        # --- YOUR CODE HERE ---
        # 实现前馈网络的前向传播

        # 1. 通过第一个线性层，然后是ReLU激活函数。
        output = self.relu(self.w_1(x))

        # 2. 通过第二个线性层。
        output = self.w_2(output)
        output = self.dropout(output)

        # 3. Add & Norm: 添加残差连接并应用 Layer Normalization。
        output = self.layer_norm(output + residual)

        return output
        # --- END YOUR CODE ---


class PositionalEncoding(nn.Module):
    """
    原生实现位置编码 (Positional Encoding)。
    使用 sin 和 cos 函数为序列中的每个位置提供唯一的位置信息。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model (int): 模型的维度。
            max_len (int, optional): 可处理的最大序列长度. Defaults to 5000.
            dropout (float, optional): Dropout概率. Defaults to 0.1.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够大的位置编码矩阵，形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # --- YOUR CODE HERE ---
        # 计算位置编码矩阵 pe

        # 1. 创建位置张量 `position`，形状为 [max_len, 1]。
        #    提示: 使用 torch.arange() 和 .unsqueeze(1)
        position = torch.arange(0, max_len).unsqueeze(1)

        # 2. 计算除法项 `div_term`。公式为: 1 / (10000^(2i/d_model))
        #    这等价于 exp(2i * (-log(10000.0) / d_model))
        #    `i` 的范围是 [0, 1, ..., d_model/2 - 1]
        #    提示: 使用 torch.arange(0, d_model, 2) 和 torch.exp()
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # 3. 为偶数索引应用 sin 函数: pe[:, 0::2] = sin(position * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 4. 为奇数索引应用 cos 函数: pe[:, 1::2] = cos(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # --- END YOUR CODE ---

        # 增加一个 batch 维度，方便后续广播
        # pe shape: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 将 pe 注册为 buffer。它不是模型参数，但希望它能随模型移动（如 .to(device)）
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入的词嵌入张量, shape [B, L, d_model]

        Returns:
            torch.Tensor: 增加了位置信息的张量, shape [B, L, d_model]
        """
        # --- YOUR CODE HERE ---
        # 将位置编码添加到输入张量 x 上。
        # 提示: x 的序列长度可能小于 max_len，所以需要对 pe进行切片。
        # self.pe 的形状是 [1, max_len, d_model]，它会自动广播到 x 的 batch size。
        # x = x + self.pe[:, :x.size(1), :]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
        # --- END YOUR CODE ---


# LayerNorm is a standard PyTorch module, but if you want students to implement it:
class LayerNorm(nn.Module):
    """
    原生实现层归一化 (Layer Normalization)。
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))  # 可学习的缩放参数
        self.beta = nn.Parameter(torch.zeros(features))  # 可学习的偏移参数
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, L, d_model]
        # --- YOUR CODE HERE ---
        # 实现层归一化的前向传播

        # 1. 在最后一个维度 (d_model) 上计算均值和方差。
        #    提示: 使用 x.mean(-1, keepdim=True) 和 x.var(-1, keepdim=True)
        mean = x.mean(-1, keepdim=True)
        std = x.var(-1, keepdim=True)

        # 2. 归一化 x。
        #    公式: (x - mean) / sqrt(var + eps)
        normalized_x = (x - mean) / torch.sqrt(std + self.eps)

        # 3. 应用可学习的参数 gamma 和 beta。
        #    公式: gamma * normalized_x + beta
        output = self.gamma * normalized_x + self.beta

        return output
        # --- END YOUR CODE ---
