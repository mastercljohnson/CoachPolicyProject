import os
import requests
import tiktoken
import numpy as np
from tqdm import tqdm

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token  # end of text token id for gpt2

# Add EOT once at the end
ids = enc.encode_ordinary(data) + [eot]  # one EOT token at the end

# Train/val split
n = len(ids)
train_ids = ids[:int(n * 0.9)]
val_ids = ids[int(n * 0.9):]

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Debug info
print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens:   {len(val_ids):,}")
print(f"Example (first 20): {train_ids[:20]}")

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens