import numpy as np
def positional_encoding_matrix(num_tokens,embed_dim):
    shape = (num_tokens,embed_dim)
    pos_matrix = np.zeros(shape,dtype=np.float32)
    for pos in range(num_tokens):
        for i in range(embed_dim):
            if i%2:
                pos_matrix[pos][i] = np.sin(pos/np.pow(10000,i/embed_dim))
            else:
                pos_matrix[pos][i] = np.cos(pos/np.pow(10000,(i-1)/embed_dim))
    return pos_matrix