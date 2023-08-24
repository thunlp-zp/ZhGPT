import torch

import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    """
    Scale Dot Product Attention

    """

    def __init__(self, d_k, dropout=0.1):
        super(ScaleDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, attn_mask):
        # q: [batch_size, n_heads, len_q, d_k]
        # k: [batch_size, n_heads, len_k, d_k]
        # v: [batch_size, n_heads, len_v, d_v]
        # attn_mask: [batch_size, n_heads, len_q, len_k]

        # [batch_size, n_heads, len_q, len_k]
        attn_score = torch.matmul(
            q, k.transpose(-1, -2)) / (self.d_k ** 0.5)
        if attn_mask is not None:
            attn_score.masked_fill_(attn_mask, -1e9)

        attn_weights = nn.Softmax(dim=-1)(attn_score)

        # [batch_size, n_heads, len_q, len_k]
        attn_weights = self.dropout(attn_weights)

        # [batch_size, n_heads, len_q, d_v]
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.attention = ScaleDotProductAttention(self.d_k, dropout=dropout)

        self.linear = nn.Linear(self.n_heads * self.d_v, d_model)

    def forward(self, q, k, v, attn_mask):
        # q: [batch_size, len_q, d_model]
        # k: [batch_size, len_k, d_model]
        # v: [batch_size, len_v, d_model]
        # attn_mask: [batch_size, len_q, len_k]
        #

        batch_size = q.size(0)

        # q_heads: [batch_size, n_heads, len_q, d_k]
        q_heads = self.W_q(q).view(
            batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.W_k(k).view(
            batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.W_v(v).view(
            batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # attn_mask: [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # output: [batch_size, n_heads, len_q, d_v]
        # attn_weights: [batch_size, n_heads, len_q, len_k]
        attn, attn_weights = self.attention(
            q_heads, k_heads, v_heads, attn_mask)

        # attn: [batch_size, len_q, n_heads * d_v]
        attn = attn.transpose_(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)

        # output: [batch_size, len_q, d_model]
        output = self.linear(attn)

        return output, attn_weights


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear2.weight, std=0.02)

    def forward(self, inputs):
        # inputs: [batch_size, len_q, d_model]

        # outputs: [batch_size, len_q, d_ff]
        outputs = self.gelu(self.linear1(inputs))

        # outputs: [batch_size, len_q, d_model]
        outputs = self.linear2(outputs)

        return outputs


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, residual_dropout):
        super(DecoderLayer, self).__init__()
        # multi-head attention
        self.multi_attention = MultiHeadAttention(
            d_model, n_heads, dropout=dropout)

        # Residual Dropout
        self.dropout = nn.Dropout(p=residual_dropout)

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-5)

        # Feed Forward Network
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)

        # Residual Dropout
        self.dropout2 = nn.Dropout(p=residual_dropout)

        # Layer Normalization
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, inputs, attn_mask):
        # inputs: [batch_size, len_q, d_model]
        # attn_mask: [batch_size, len_q, len_k]

        # multi-head attention
        # attn: [batch_size, len_q, d_model]
        attn, attn_weights = self.multi_attention(
            inputs, inputs, inputs, attn_mask)

        # Residual Dropout
        attn = self.dropout(attn)

        # Layer Normalization
        attn = self.layer_norm1(inputs + attn)

        # Feed Forward Network
        # ffn_outputs : [batch_size, len_q, d_model]
        ffn_outputs = self.ffn(attn)

        # Residual Dropout
        ffn_outputs = self.dropout2(ffn_outputs)

        # Layer Normalization
        ffn_outputs = self.layer_norm2(ffn_outputs + attn)

        return ffn_outputs, attn_weights


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, n_layers, n_heads, d_ff, dropout, residual_dropout, embed_dropout, pad_idx):
        super(TransformerDecoder, self).__init__()

        self.pad_idx = pad_idx

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.dropout = nn.Dropout(p=embed_dropout)
        self.position_embedding = nn.Embedding(seq_len + 1, d_model)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout, residual_dropout)
             for _ in range(n_layers)]
        )

        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, inputs):
        # inputs: [batch_size, len_q]
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(
            inputs.size(0), inputs.size(1)) + 1
        position_pad_mask = inputs.eq(self.pad_idx)
        # positions: [batch_size, len_q]
        positions.masked_fill_(position_pad_mask, 0)

        # outputs: [batch_size, len_q, d_model]
        outputs = self.dropout(self.embedding(
            inputs) + self.position_embedding(positions))
        attn_pad_mask = self.get_attention_mask(inputs, inputs, self.pad_idx)

        subsequent_mask = self.get_attention_subsequent_mask(
            inputs).to(device=attn_pad_mask.device)

        attn_mask = torch.gt(
            (attn_pad_mask.to(device=attn_pad_mask.device) + subsequent_mask), 0)

        attention_weights = []
        for layer in self.layers:
            # outputs: [batch_size, len_q, d_model]
            # attn_weights: [batch_size, n_heads, len_q, len_k]
            outputs, attn_weights = layer(outputs, attn_mask)

            attention_weights.append(attn_weights)

        return outputs, attention_weights=

    def get_attention_mask(self, q, k, pad_idx):
        # In this function, we will get a attention mask, the shape of the mask is [batch_size, len_q, len_k], will not consider the <pad> token
        attn_pad_mask = k.eq(pad_idx).unsqueeze(1).repeat(1, q.size(1), 1)
        return attn_pad_mask

    def get_attention_subsequent_mask(self, q):
        # 返回了一个上三角矩阵，偏移量为1，不考虑<bos_token>
        batch_size, q_len = q.size()
        subsequent_mask = torch.ones(
            batch_size, q_len, q_len, dtype=q.dtype).triu(diagonal=1)
        return subsequent_mask


class GPT1(nn.Module):
    def __init__(self, vocab_size,
                 d_model,
                 seq_len,
                 n_layers, 
                 n_heads, 
                 d_ff, 
                 dropout, 
                 residual_dropout, 
                 embed_dropout, 
                 pad_idx):
        super(GPT1, self).__init__()

        self.decoder = TransformerDecoder(vocab_size, 
                                          d_model, 
                                          seq_len, 
                                          n_layers, 
                                          n_heads, 
                                          d_ff, 
                                          dropout, 
                       