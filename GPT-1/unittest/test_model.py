import torch
import sys
sys.path.append("/home/zhoupeng/zh2nlp_expr/ZhGPT/GPT-1/")

from model import (PositionWiseFeedForwardNetwork,
                   MultiHeadAttention,
                   ScaleDotProductAttention,
                   TransformerDecoder,
                   DecoderLayer,
                   GPT1,
                   GPT1LMHead)

def model_test(schema, **args):
    def test_scale_dot_product_attention():
        sdpa = schema(d_k=args["d_k"], dropout=0.1)
        query = torch.randn(args["batch_size"],
                            args["n_heads"], args["seq_len"], args["d_k"])
        key = torch.randn(args["batch_size"],
                          args["n_heads"], args["seq_len"], args["d_k"])
        value = torch.randn(args["batch_size"],
                            args["n_heads"], args["seq_len"], args["d_k"])
        mask = torch.randn(
            args["batch_size"], args["n_heads"], args["seq_len"], args["seq_len"])
        outputs, attn = sdpa(query, key, value, attn_mask=None)
        print(outputs)

    def test_position_wise_feed_forward_network():
        model = schema(args)
        print(model)

    def test_multi_head_attention():
        model = schema(512, 8)
        print(model)

    def test_decoder_layer():
        model = schema(512, 8, 2048, 0.1, 0.1)
        print(model)

    def test_transformer_decoder():
        model = schema(vocab_size=args["vocab_size"], 
                                   d_model=args["d_model"], 
                                   seq_len=args["seq_len"], 
                                   n_layers=args["n_layers"], 
                                   n_heads=args["n_heads"], 
                                   d_ff=args["d_ff"], 
                                   dropout=args["dropout"], 
                                   residual_dropout=args["residual_dropout"], 
                                   embed_dropout=args["embed_dropout"], 
                                   pad_idx=args["pad_idx"])
        # now, we will create a batch of inputs, the shape of inputs is [batch_size, seq_len]
        # we can use torch.randint to generate a batch of random integers, the first argument is low, the second argument is high, the third argument is the shape of the output
        inputs = torch.randint(0, args["vocab_size"], (args["batch_size"], args["seq_len"]))
        outputs, attention_weights = model(inputs)

    def test_gpt1():
        model = schema(30000, 512, 100, 6, 8, 2048, 0.1, 0.1, 0.1, 1)
        print(model)

    def test_gpt1_lm_head():
        model = schema(30000, 512, 100, 6, 8, 2048, 0.1, 0.1, 0.1, 1)
        print(model)

    if schema.__name__ == "PositionWiseFeedForwardNetwork":
        test_position_wise_feed_forward_network()
    elif schema.__name__ == "MultiHeadAttention":
        test_multi_head_attention()
    elif schema.__name__ == "ScaleDotProductAttention":
        test_scale_dot_product_attention()
    elif schema.__name__ == "DecoderLayer":
        test_decoder_layer()
    elif schema.__name__ == "TransformerDecoder":
        test_transformer_decoder()
    elif schema.__name__ == "GPT1":
        test_gpt1()
    elif schema.__name__ == "GPT1LMHead":
        test_gpt1_lm_head()
    else:
        raise ValueError("Unknown schema: {}".format(schema))


if __name__ == "__main__":
    vocab_size = 10
    batch_size = 1
    d_model = 16
    seq_len = 10
    n_layers = 2
    n_heads = 2
    d_ff = 30
    dropout = 0.1
    residual_dropout = 0.1
    embed_dropout = 0.1
    pad_idx = 0
    assert d_model % n_heads == 0, "d_model % n_heads != 0"
    d_k = d_v = d_model // n_heads
    # model_test(schema=ScaleDotProductAttention, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, seq_len=seq_len, batch_size=batch_size)
    model_test(schema=TransformerDecoder, batch_size=batch_size, vocab_size=vocab_size, d_model=d_model, seq_len=seq_len, n_layers=n_layers,
               n_heads=n_heads, d_ff=d_ff, dropout=dropout, residual_dropout=residual_dropout, embed_dropout=embed_dropout, pad_idx=pad_idx)
    # model_test(schema=GPT1, vocab_size=vocab_size, d_model=d_model, seq_len=seq_len, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout, residual_dropout=residual_dropout, embed_dropout=embed_dropout, pad_idx=pad_idx)
    # model_test(schema=DecoderLayer, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, residual_dropout=residual_dropout)
    # model_test(schema=MultiHeadAttention, d_model=d_model, n_heads=n_heads)

    # model_test(schema=PositionWiseFeedForwardNetwork, d_model=d_model, d_ff=d_ff)
    # model_test(schema=GPT1LMHead, vocab_size=vocab_size, d_model=d_model, seq_len=seq_len, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout, residual_dropout=residual_dropout, embed_dropout=embed_dropout, pad_idx=pad_idx)
