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


        # Q, K, V are [B, T, d_model] -> transform -> [B, T, d_k]
        x = torch.matmul(Q, K.transpose(-2, -1))
        x = x = x / float(self.key_dim ** 0.5) # Scale the attention scores by the square root of the key dimension
        # Note we set the upper triangular part to -inf because 
        # we are viewing these as row vectors and the causal relation in the columns
        T= Q.size(1)
        tril = torch.tril(torch.ones(T, T, device=Q.device))# Create a lower triangular mask
        mask = tril.unsqueeze(0).expand(x.size(0), -1, -1)
        # x = x.masked_fill(tril == 0, float('-inf')) # Mask out the padding tokens
        x = x.masked_fill(mask == 0, float('-inf')) # Mask out the padding tokens
        x_scaled = nn.functional.softmax(x,dim=-1)

        x_scaled = self.attn_dropout(x_scaled) # Apply dropout to the attention scores

        attention_head_output = torch.matmul(x_scaled, V)
        return attention_head_output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_input_tokens, encode_dim, key_dim, value_dim,dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.encode_dim = encode_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Initialize the attention heads
        # encode dimensions is split among heads to detect different things
        self.attention_heads = nn.ModuleList([
            AttentionHead(num_input_tokens, encode_dim, key_dim, value_dim,dropout) for _ in range(num_heads)
        ])

        self.linear = nn.Linear(num_heads * value_dim, encode_dim) # {h, d_v} -> {h, d_model}

    def forward(self, K, Q, V):
        # B_k, T_k, C_k = K.size()
        # B_q, T_q, C_q = Q.size()
        # B_v, T_v, C_v = V.size()
        outputs = [head(K, Q, V) for head in self.attention_heads]
        concat_outputs = torch.cat(outputs, dim=-1) # Concatenate along the last dimension (B, T, n_head * output_dim)
        return self.linear(concat_outputs)
    
class FeedForward(nn.Module):
    def __init__(self, encode_dim,dropout=0.2):
        super().__init__()
        self.encode_dim = encode_dim # d_model
        self.network = nn.Sequential(
            nn.Linear(encode_dim, 4 * encode_dim), # {h, d_model} -> {h, 4 * d_model}
            nn.GELU(), # Change from relu to gelu activation
            nn.Linear(4 * encode_dim, encode_dim), # {h, 4 * d_model} -> {h, d_model}
            nn.Dropout(dropout) # Apply dropout
        )

    def forward(self, x):
        x = self.network(x) # Pass through the feedforward network
        return x

# TODO:Check again if this is correct.
class TransformerBlock(nn.Module):
    def __init__(self, num_heads, num_input_tokens, encode_dim, key_dim, value_dim,dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.encode_dim = encode_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Initialize multi-head attention and feedforward network
        self.multi_head_attention = MultiHeadAttention(num_heads, num_input_tokens, encode_dim, key_dim, value_dim,dropout)
        self.feed_forward = FeedForward(encode_dim, dropout) # Feedforward network with dropout

        # Initialize layer normalization
        self.layernorm_1 = nn.LayerNorm(encode_dim) # Layer normalization
        self.layernorm_2 = nn.LayerNorm(encode_dim) # Layer normalization

    def forward(self, input):
        # Apply multi-head attention
        # TODO: Warning, not sure if i need to deep copy the input here.
        layer_norm_1_output = self.layernorm_1(input)  # Apply layer normalization before attention

        attention_output = self.multi_head_attention(layer_norm_1_output, layer_norm_1_output, layer_norm_1_output) # K, Q, V are all the same in this case

        # Add residual connection and apply layer normalization
        residual_1_output = input + attention_output # Pass through the feedforward network

        layer_norm_2_output = self.layernorm_2(residual_1_output)  # Apply layer normalization after attention

        # Apply second layer normalization
        output = residual_1_output + self.feed_forward(layer_norm_2_output)
        
        return output
    
class TransformerDecoder(nn.Module):
    def __init__(self, num_heads, num_input_tokens, encode_dim, key_dim, value_dim, num_blocks, vocab_size=50304,dropout=0.2):
        super().__init__()
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.num_input_tokens = num_input_tokens  # Number of input tokens
        self.embedding = nn.Embedding(vocab_size, encode_dim)  # Embedding layer to convert input tokens to vectors of size encode_dim
        self.dropout = nn.Dropout(dropout)  # Dropout layer to prevent overfitting
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(num_heads, num_input_tokens, encode_dim, key_dim, value_dim) for _ in range(num_blocks)
        ])

        pe = torch.from_numpy(positional_encoding_matrix(num_input_tokens, encode_dim)).float()
        self.register_buffer("positional_encoding", pe)  # Register positional encoding as a buffer to avoid it being treated as a parameter

        self.final_ln = nn.LayerNorm(encode_dim) # Final layer normalization before the output projection

        # self.linear = nn.Linear(encode_dim, vocab_size, bias=False) # {h, d_model} -> {h, vocab_size}
        self.output_projection = nn.Linear(encode_dim, vocab_size, bias=False)  # Placeholder; we tie it below
        # self.output_projection.weight = self.embedding.weight

    # Residual connection in transformer preserves positional encoding
    def forward(self, input, targets=None, label_smoothing=0.0):
        input = input.to(torch.long)  # Ensure input is of type long for embedding lookup
        input = self.embedding(input)  # Convert input tokens to vectors
        B, T, _ = input.shape
        input = input + 0.1 * self.positional_encoding[:T, :].unsqueeze(0)  # Add positional encoding to the input
        input = self.dropout(input)  # Apply dropout to the input embeddings
        for block in self.transformer_blocks:
            input = block(input)
        
        #  Apply layernorm before final layer projection, i think this is the final fix?
        input = self.final_ln(input)  # Apply final layer normalization

        # linear_output = self.linear(input)  # Apply linear transformation to the output of the last transformer block
        linear_output = self.output_projection(input)  # Project to vocabulary size

        if targets is not None:
            loss = nn.functional.cross_entropy(linear_output.view(-1, linear_output.size(-1)), targets.reshape(-1), ignore_index=-1, label_smoothing=label_smoothing)      
        else:
            loss = None
        
        return linear_output, loss
    
    @torch.no_grad()
    def generate(self, input, max_length=20):
        for _ in range(max_length):
            cropped_input = input[:, -self.num_input_tokens:]  # Ensure input is within the max length
            logits, _ = self.forward(cropped_input)
            probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)  # Get probabilities for the last token
            next_token = torch.multinomial(probs, num_samples=1)
            input = torch.cat([input, next_token], dim=1)
        return input
    
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