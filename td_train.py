#  Use bpe to train transformer decoder model on tiny shakespeare dataset
import os
import torch
import numpy as np
from transformer_decoder import TransformerDecoder
from contextlib import nullcontext
import torch_directml
import math
# from shakespeare_prep.prepare import train_ids, val_ids

# block_size = 256  # Input size
block_size = 30  # Input size

# poor man's data loader
data_dir = os.path.join('shakespeare_prep')
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    x, y = x.to(device), y.to(device)
    
    return x, y


eval_iters = 200

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # with ctx: # Removing this line because directml does not support autocast embedding?
            logits, loss = model(X, Y)  
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    

    # Set device
    device_type = 'privateuseone'  # Use 'cpu' for CPU, 'cuda' for nvidia GPU, or 'dml' for DirectML
    # ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.float32)
    # device = torch.device("cpu")
    device = torch_directml.device()  # Use DirectML for GPU acceleration
    print(f"Using device: {device}")

    # Transformer Hyperparameters from https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py
    num_heads = 4
    num_blocks = 4  # Number of transformer blocks
    # num_input_tokens = 256  # Input size
    num_input_tokens = 30  # Input size
    encode_dim = 384  # Example encoding dimension
    key_dim = 64  # Example key dimension
    value_dim = 64  # Example value dimension

    # Training params
    # batch_size = 64  # Example batch size
    batch_size = 5  # Example batch size
    learning_rate = 1e-3 # with baby networks can afford to go a bit higher
    max_iters = 2000
    lr_decay_iters = 2000 # make equal to max_iters usually
    warmup_iters = 100 # usually 0.1 * lr_decay_iters
    min_lr = 1e-4 # learning_rate / 10 usually
    beta1 = 0.9 # momentum term for Adam optimizer
    beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
    weight_decay = 1e-1 # L2 regularization term

    # learning rate decay scheduler (cosine with warmup)
    #  See if validation loss decreases after scheduler implemented
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    gradient_accumulation_steps = 10 # used to simulate larger batch sizes

    # warmup_iters = 100

    vocab_size = 50304  # Size of the vocabulary, e.g., number of unique characters in the dataset
    

    # Initialize the transformer decoder model
    model = TransformerDecoder(num_heads, num_input_tokens, encode_dim, key_dim, value_dim, num_blocks, vocab_size).to(device)

    # TODO: optimizer, are the right model parameters being optimized? Check
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Training loop (simplified)
    X,Y = get_batch('train')  # Get a batch of training data
    # print(X,Y)
    # model.train()
    iter_num = 0
    # eval_interval = 2000
    eval_interval = 20
    best_eval_loss = float('inf')  # Initialize best validation loss
    while True:

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(iter_num)

        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_eval_loss:
                best_eval_loss = losses['val']
                print(f"New best validation loss: {best_eval_loss:.4f} at step {iter_num}")
                checkpoint = model.state_dict()
                torch.save(checkpoint, f"./shakespeare_checkpoints/shakespeare_{iter_num}.pth")
            
        
        for micro_step in range(gradient_accumulation_steps):
            # print(X.shape, Y.shape)
            logits, loss = model(X, Y)
            # print("gets loss and predict")
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

            # print(f"Average loss over gradient accumulation steps {loss.item():.4f}")
            
            X, Y = get_batch('train')

            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            optimizer.zero_grad()
        
        iter_num += 1

        if iter_num >= max_iters:
            print(f"Training complete after {iter_num} iterations.")
            break