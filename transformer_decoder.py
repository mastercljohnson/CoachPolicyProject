import torch
from torch import nn
from positional_encoding import positional_encoding_matrix


# Implementation of a Transformer Decoder with Multi-Head Attention and Feed Forward Network
class AttentionHead(nn.Module):
    def __init__(self, num_input_tokens, encode_dim, key_dim, value_dim, dropout=0.2):
        super().__init__()
        self.num_input_tokens = num_input_tokens # of input tokens
        self.encode_dim = encode_dim # d_model
        self.key_dim = key_dim #d_k
        self.value_dim = value_dim # d_v

        # For this implementation for self understanding, choose not to multiply by # of heads
        self.keys = nn.Linear(encode_dim,key_dim,bias=False) # {h, d_model} -> {h, d_k}
        self.queries = nn.Linear(encode_dim,key_dim,bias=False) # {h, d_model} -> {h, d_k}
        self.values = nn.Linear(encode_dim,value_dim,bias=False) # {h, d_model} -> {h, d_v}

        # Dropout layer for attention
        self.attn_dropout = nn.Dropout(dropout)
        
        

    def forward(self, K, Q, V):
        # K, Q, V are the keys, queries and values respectively.
        K = self.keys(K) # {h, d_model} -> {h, d_k}
        Q = self.queries(Q) # {h, d_model} -> {h, d_k}
        V = self.values(V) # {h, d_model} -> {h, d_v}
        # print(f"K Q V {K.shape}, {Q.shape}, {V.shape}")


        x = torch.matmul(Q, K.transpose(-2, -1))
        # Note we set the upper triangular part to -inf because 
        # we are viewing these as row vectors and the causal relation in the columns
        tril = torch.tril(torch.ones(self.num_input_tokens, self.num_input_tokens)).to('privateuseone')# Create a lower triangular mask
        x = x.masked_fill(tril == 0, float('-inf')) # Mask out the padding tokens
        x_scaled = torch.div(x, self.key_dim**0.5)
        x_scaled = nn.functional.softmax(x_scaled,dim=-1)

        x_scaled = self.attn_dropout(x_scaled) # Apply dropout to the attention scores

        attention_head_output = torch.matmul(x_scaled, V)
        return attention_head_output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_input_tokens, encode_dim, key_dim, value_dim):
        super().__init__()
        self.num_heads = num_heads
        self.encode_dim = encode_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        

        # Initialize the attention heads
        # encode dimensions is split among heads to detect different things
        self.attention_heads = nn.ModuleList([
            AttentionHead(num_input_tokens, encode_dim, key_dim, value_dim) for _ in range(num_heads)
        ])

        self.linear = nn.Linear(num_heads * value_dim, encode_dim) # {h, d_v} -> {h, d_model}

    def forward(self, K, Q, V):
        
        B_k, T_k, C_k = K.size()
        B_q, T_q, C_q = Q.size()
        B_v, T_v, C_v = V.size()
        # print(f"key B T C {B_k}, {T_k}, {C_k} ")
        # print(f"query B T C {B_q}, {T_q}, {C_q}")
        # print(f"value B T C {B_v}, {T_v}, {C_v}")


        outputs = [head(K, Q, V) for head in self.attention_heads]
        concat_outputs = torch.cat(outputs, dim=-1) # Concatenate along the last dimension (B, T, n_head * output_dim)
        return self.linear(concat_outputs)
    
class FeedForward(nn.Module):
    def __init__(self, encode_dim,dropout=0.2):
        super().__init__()
        self.encode_dim = encode_dim # d_model
        self.network = nn.Sequential(
            nn.Linear(encode_dim, 4 * encode_dim), # {h, d_model} -> {h, 4 * d_model}
            nn.ReLU(), # Apply ReLU activation
            nn.Linear(4 * encode_dim, encode_dim), # {h, 4 * d_model} -> {h, d_model}
            nn.Dropout(dropout) # Apply dropout
        )

    def forward(self, x):
        x = self.network(x) # Pass through the feedforward network
        return x

# TODO:Check again if this is correct.
class TransformerBlock(nn.Module):
    def __init__(self, num_heads, num_input_tokens, encode_dim, key_dim, value_dim):
        super().__init__()
        self.num_heads = num_heads
        self.encode_dim = encode_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Initialize multi-head attention and feedforward network
        self.multi_head_attention = MultiHeadAttention(num_heads, num_input_tokens, encode_dim, key_dim, value_dim)
        self.feed_forward = FeedForward(encode_dim)

        # Initialize layer normalization
        self.layernorm_1 = nn.LayerNorm(encode_dim) # Layer normalization
        self.layernorm_2 = nn.LayerNorm(encode_dim) # Layer normalization

    def forward(self, input):
        # Apply multi-head attention
        # TODO: Warning, not sure if i need to deep copy the input here.
        attention_output = self.multi_head_attention(input, input, input) # K, Q, V are all the same in this case

        # Add residual connection and apply layer normalization
        layer_norm_1_output = input + self.layernorm_1(attention_output)  # Pass through the feedforward network

        feed_forward_output = self.feed_forward(layer_norm_1_output)

        # Apply second layer normalization
        output = layer_norm_1_output + self.layernorm_2(feed_forward_output)
        
        return output
    
class TransformerDecoder(nn.Module):
    def __init__(self, num_heads, num_input_tokens, encode_dim, key_dim, value_dim, num_blocks, vocab_size=50304,dropout=0.2):
        super().__init__()
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, encode_dim)  # Embedding layer to convert input tokens to vectors of size encode_dim
        self.dropout = nn.Dropout(dropout)  # Dropout layer to prevent overfitting
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(num_heads, num_input_tokens, encode_dim, key_dim, value_dim) for _ in range(num_blocks)
        ])
        self.positional_encoding = torch.from_numpy(positional_encoding_matrix(num_input_tokens, encode_dim)).to('privateuseone')
        # print(self.positional_encoding.dtype) float32

        self.linear = nn.Linear(encode_dim, vocab_size, bias=False) # {h, d_model} -> {h, vocab_size}

    # Residual connection in transformer preserves positional encoding
    def forward(self, input, targets=None):
        input = self.embedding(input)  # Convert input tokens to vectors
        # print(f"input shape {input.shape}")
        # print(f"positional encoding shape {self.positional_encoding.shape}")
        input = input + self.positional_encoding
        # print("Successfully added positional encoding")
        input = self.dropout(input)  # Apply dropout to the input embeddings
        for block in self.transformer_blocks:
            input = block(input)
            # print("hrnggggggg")
        linear_output = self.linear(input)  # Apply linear transformation to the output of the last transformer block
        # print("REACHES HERE")
        if targets is not None:
            # If targets are provided, compute the loss
            # print(f"linear output shape {linear_output.shape}")
            # print(f"targets shape {targets.shape}")

            # for i in range(targets.shape[0]):
            #     if targets[i].max() >= self.vocab_size:
            #         print(f"Warning: target {targets[i]} has values >= vocab_size {self.vocab_size}")
            
            # print("Finish checking target vocab size")

            loss = nn.functional.cross_entropy(linear_output.view(-1, linear_output.size(-1)), targets.view(-1), ignore_index=-1)
            # print(f"loss {loss}")
        else:
            loss = None
        # return nn.functional.softmax(linear_output, dim=-1)
        
        return linear_output, loss
    
# if __name__ == "__main__":
#     # Define the layers of the neural network and initialize weights and biases
#     vocab_size = 3
#     num_input_tokens = 5
#     encode_dim = 10
#     key_dim = 3
#     value_dim = 4
#     num_heads = 3
#     num_blocks = 2
#     batch_size = 7

#     # Transformer Block test
#     td = TransformerDecoder(num_heads, num_input_tokens, encode_dim, key_dim, value_dim,num_blocks)

#     # Transformer test
    
#     # Create a random input tensor
#     key_tensor = torch.randn(batch_size, num_input_tokens, encode_dim)
#     # print(key_tensor.dtype)

#     td_out = td(key_tensor)
#     print("Transformer output shape:", td_out.shape)