import math
import torch
from torch import nn
import lightning as pl

# 定义Self-Attention模块
# 1. Q K V <--- linear
# 2. Q x K^T ---> attn_weight
# 3. attn_weight <--- causal make <--- softmax <--- dropout	
# 4. attn_weight x V ---> output
# 5. output <--- linear
class SelfAttention(pl.LightningModule):
	def __init__(self, embed_dim: int, attention_dim: int, dropout: float = 0.1):
		super().__init__()
		self.attention_dim = attention_dim
		self.query = nn.Linear(embed_dim, embed_dim)
		self.key = nn.Linear(embed_dim, embed_dim)
		self.value = nn.Linear(embed_dim, embed_dim)

		self.attn_scale = math.sqrt(attention_dim)

		self.dropout = nn.Dropout(dropout)

		self.out_layer = nn.Linear(embed_dim, attention_dim)

	def forward(self, x, mask=None):
		# x shape: (batch_size, seq_len, embed_dim)
		# 计算Q, K, V
		Q = self.query(x)  # Q.shape is (batch_size, seq_len, embed_dim)
		K = self.key(x)    # K.shape is (batch_size, seq_len, embed_dim)
		V = self.value(x)  # V.shape is (batch_size, seq_len, embed_dim)

		attn_scores = Q @ K.transpose(-1, -2) / self.attn_scale  # (batch_size, seq_len, seq_len)
		if mask is not None:
			attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
		attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, seq_len, seq_len)
		attn_weights = self.dropout(attn_weights)

		# 计算加权值
		result = attn_weights @ V  # (batch_size, seq_len, embed_dim)

		# 最终线性变换
		result = self.out_layer(result)  # (batch_size, seq_len, attention_dim)

		return result

# 定义Multi-Head Self-Attention模块
class MultiHeadAttention(pl.LightningModule):
	def __init__(self,  embed_dim: int, num_heads: int, dropout: float = 0.1):
		super().__init__()
		self.num_heads = num_heads
		self.attention_dim = embed_dim // num_heads
		assert self.attention_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
		self.query = nn.Linear(embed_dim, self.attention_dim * num_heads)
		self.key = nn.Linear(embed_dim, self.attention_dim * num_heads)
		self.value = nn.Linear(embed_dim, self.attention_dim * num_heads)

		self.attn_scale = math.sqrt(self.attention_dim)

		self.dropout = nn.Dropout(dropout)

		self.out_layer = nn.Linear(self.attention_dim * num_heads, embed_dim)

	def forward(self, x, mask=None):
		# x shape: (batch_size, seq_len, embed_dim)
		bs, seq_len, _ = x.size()
		# 计算Q, K, V
		Q = self.query(x)  # Q.shape is (batch_size, seq_len, attention_dim * num_heads)
		K = self.key(x)    # K.shape is (batch_size, seq_len, attention_dim * num_heads)
		V = self.value(x)  # V.shape is (batch_size, seq_len, attention_dim * num_heads)

		# 分割头
		Q = Q.view(bs, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, attention_dim)
		K = K.view(bs, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, attention_dim)
		V = V.view(bs, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, attention_dim)

		# 计算注意力权重
		attn_scores = Q @ K.transpose(-1, -2) / self.attn_scale  # (batch_size, num_heads, seq_len, seq_len)
		if mask is not None:
			attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
		attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
		attn_weights = self.dropout(attn_weights)

		# 计算加权值
		result = attn_weights @ V  # (batch_size, num_heads, seq_len, attention_dim)
		result = result.transpose(1, 2).contiguous().view(bs, seq_len, self.attention_dim * self.num_heads)  # (batch_size, seq_len, attention_dim * num_heads)

		# 最终线性变换
		result = self.out_layer(result)  # (batch_size, seq_len, embed_dim)

		return result

# 定义Cross-Attention模块
class CrossAttention(pl.LightningModule):
	def __init__(self, embed_dim:int, num_heads:int, dropout:float=0.1):
		super().__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.attention_dim = embed_dim // num_heads

		assert self.attention_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

		self.query = nn.Linear(embed_dim, self.attention_dim * num_heads)
		self.key = nn.Linear(embed_dim, self.attention_dim * num_heads)
		self.value = nn.Linear(embed_dim, self.attention_dim * num_heads)

		self.attn_scale = math.sqrt(self.attention_dim)

		self.dropout = nn.Dropout(dropout)

		self.out_layer = nn.Linear(self.attention_dim * num_heads, embed_dim)

	def forward(self, query, key_value, mask=None):
		bs, query_len, _ = query.size()
		_, kv_len, _ = key_value.size()

		# 计算Q, K, V
		Q = self.query(query)  # Q.shape is (batch_size, seq_len, attention_dim * num_heads)
		K = self.key(key_value)    # K.shape is (batch_size, seq_len, attention_dim * num_heads)
		V = self.value(key_value)  # V.shape is (batch_size, seq_len, attention_dim * num_heads)

		# 分割头
		Q = Q.view(bs, query_len, self.num_heads, self.attention_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, attention_dim)
		K = K.view(bs, kv_len, self.num_heads, self.attention_dim).transpose(1, 2)  # (batch_size, num_heads, kv_len, attention_dim)
		V = V.view(bs, kv_len, self.num_heads, self.attention_dim).transpose(1, 2)  # (batch_size, num_heads, kv_len, attention_dim)

		# 计算注意力权重
		attn_scores = Q @ K.transpose(-1, -2) / self.attn_scale  # (batch_size, num_heads, query_len, kv_len)
		attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, query_len, kv_len)
		attn_weights = self.dropout(attn_weights)

		# 计算加权值
		result = attn_weights @ V  # (batch_size, num_heads, query_len, attention_dim)
		result = result.transpose(1, 2).contiguous().view(bs, query_len, self.attention_dim * self.num_heads)  # (batch_size, query_len, attention_dim * num_heads)

		# 最终线性变换
		result = self.out_layer(result)  # (batch_size, query_len, embed_dim)

		return result
