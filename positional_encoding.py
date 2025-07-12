import numpy as np
# def positional_encoding_matrix(num_tokens,embed_dim):
#     shape = (num_tokens,embed_dim)
#     pos_matrix = np.zeros(shape,dtype=np.float32)
#     for pos in range(num_tokens):
#         for i in range(embed_dim):
#             if i%2:
#                 pos_matrix[pos][i] = np.cos(pos/np.pow(10000,(i-1)/embed_dim))
#             else:
#                 pos_matrix[pos][i] = np.sin(pos/np.pow(10000,i/embed_dim))
                
#     return pos_matrix

def positional_encoding_matrix(num_tokens, embed_dim):
    pos = np.arange(num_tokens)[:, np.newaxis]
    i = np.arange(embed_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / embed_dim)
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads.astype(np.float32)
