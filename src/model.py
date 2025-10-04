import torch
import torch.nn as nn
from typing import Dict
import config  # 正确引用config

class KundokuModel(nn.Module):
    """汉文训读模型：双任务（序号预测+假名预测）- 无None类型错误"""
    def __init__(self, 
                 vocab_size: int = config.VOCAB_SIZE,
                 embed_dim: int = config.EMBED_DIM,
                 hidden_dim: int = config.HIDDEN_DIM,
                 order_num_tags: int = 10,  # 默认为10（实际训练时会覆盖）
                 kana_num_tags: int = 50) -> None:  # 默认为50（实际训练时会覆盖）
        super().__init__()
        # 1. 汉字嵌入层（忽略PAD=0）
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        
        # 2. LSTM层（CPU优化：单层+禁用Dropout）
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,          # 单层LSTM（CPU高效）
            batch_first=True,      # 输入格式：(batch_size, seq_len, embed_dim)
            dropout=0,             # 禁用Dropout（消除警告）
            bidirectional=True     # 双向LSTM，捕捉上下文
        )
        
        # 3. 双任务输出头（LSTM输出维度=2*hidden_dim，因双向）
        lstm_output_dim = 2 * hidden_dim
        self.order_head = nn.Linear(lstm_output_dim, order_num_tags)  # 序号预测
        self.kana_head = nn.Linear(lstm_output_dim, kana_num_tags)    # 假名预测

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播：输入汉字ID → 输出双任务logits"""
        # 1. 嵌入层：(batch_size, seq_len) → (batch_size, seq_len, embed_dim)
        embeddings = self.embedding(input_ids)
        
        # 2. LSTM层：输出 → (batch_size, seq_len, 2*hidden_dim)
        lstm_output, _ = self.lstm(embeddings)  # 忽略隐藏状态
        
        # 3. 双任务预测（输出logits，未经过softmax）
        order_logits = self.order_head(lstm_output)  # (batch_size, seq_len, order_num_tags)
        kana_logits = self.kana_head(lstm_output)    # (batch_size, seq_len, kana_num_tags)
        
        return {
            "order_logits": order_logits,
            "kana_logits": kana_logits
        }