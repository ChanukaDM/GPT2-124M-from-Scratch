from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import numpy as np



#-------------------------------------------

"""Multi Head Attention (instead of writing seperate head and multi head function 
we can implement both on a single class not like in previous nanoGPT"""
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #Q,K,V for all heads but in a single batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # 3x is for Q,K and V's
        #Output projections 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        #Below 2 are for the purpose of regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #Mask not a bias, where this is used in decoder to generate output based on previous context values no future values in training
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1, config.block_size,config.block_size))
        """Buffers are ideal for storing tensors that are part of the model's functionality
        but not be optimized during the training(this is inherited from nn.Module as we can even set
        persistance to false not to include this to model weigths which is a output from state_dict"""

    def forward(self,x):
        B,T,C = x.size()
        """Batch size, Sequence length,embedding dimensions(n_embd)
        calculate Q,K,V for all heads in batch and move forward to be the batch
        nh is number of heads, hs is the head size , C (number of channels) = hd * ns
        GPT2-124M : n_head=12, hs=64, so C=768 channels
        """
        qkv = self.c_attn(x)  #ex n_emb:384 , qkv output = (384, 1152)
        q, k, v = qkv.split(self.n_embd, dim=2) # split to 3 of 384
        k = k.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        q = q.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        v = v.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B,nh,T,hs)

        """
        #Multi Head attention (GPT-2 Approach) Computationally slow compared to flash attension
        att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))       1. Matmul
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('inf'))     2. Mask
        att.F.softmax(att,dim=1)                                           3. Softmax
        att = nn.dropout(att,keep_prob=0.8)                                4. Droupout
        y = att @ v # (B,nh,T,T) x (B,nh,T,hs) -> (B,nh,T,hs)              5. Matmul     """ 


        # Flash attention calculation   All above 4 steps can be done in a single step  (Fused Kernel)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y




#MLP is mostly used as the final layer of a Trasformer block
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self,x):
        x = self.c_fc(x) # ex: (384, (4*384=1536)) 
        x = self.gelu(x) # this solves vanishing gradient problem rather than using a relu
        x = self.c_proj(x) # out: (384,384) dot product calculation
        return x


#Create Blocks that require to sequentially run the machanism
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Change from += to proper residual connections
        x = x + self.attn(self.ln_1(x))  # Changed from x += self.ln_1(x)
        x = x + self.mlp(self.ln_2(x))   # Changed from x += self.mlp(x)
        return x
    




@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges with 256 byte tokens and 1 <|endoftext|>
    n_layer: int = 12 # number of layers Blocks
    n_head: int = 12 # number of heads
    n_embd: int = 768  # embeddings
    epochs = 19073



class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        """nn.ModuleDict Help us index into the sub modules
        using keys"""
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        #last linear where we apply softmax for output
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)

        #Weight sharing scheme (in GPT2 paper ) embedding layer weighths  should equal to the Linear layer before softmax
        self.transformer.wte.weight = self.lm_head.weight
        #So now we are left with a single tensor and its gonna be used twice in the forward pass

    def forward(self,idx, targets = None): # idx which is the shape of (B,T) , y 
        B,T = idx.size()
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T}"

        #forward the token and positional embeddings
        pos_shape = torch.arange(0,T, dtype=torch.long, device=idx.device) # shape of T
        posEmb = self.transformer.wte(pos_shape) #Positional Embeddings shape = (T,n_embd)
        tokEmb = self.transformer.wte(idx) # token Embeddings shape = (B,T,n_embd)
        x = tokEmb + posEmb # final input goes in to the encoder (or decoder)

        #forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        #forward the final layer norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B,T,Vocab_size)
        loss = None

        if targets is not None: 
            x_flatten = logits.view(-1, logits.size(-1)) #note that f.crossentry cant take (B,T,C) instead we flatten out to 2d (B*T, C) 
            #print("Flatten Logits: ", x_flatten.shape)
            loss = F.cross_entropy(x_flatten, targets.view(-1), reduction='mean')  
        return logits, loss # final output 




    @classmethod #class method converts a function to be a class method
    #Load GPT2 original parameters for the future model comparison
    def from_pretrained(cls,model_type):
        #loads pretrained GPT-2 model weights from huggingface
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'} # different GPT models
        from transformers import GPT2LMHeadModel
        print("LOADING WEIGHTS FROM PRETRAINED GPT: %s" % model_type)

        #n_layers,n_head and n_emb are determined from the model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  #124M parameter model
            'gpt2-medium': dict(n_layer=12, n_head=12, n_embd=1024),  
            'gpt2-large': dict(n_layer=12, n_head=12, n_embd=1280),  
            'gpt2-xl': dict(n_layer=12, n_head=12, n_embd=1600),  
        }[model_type] # get only the required model type
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024  # look back 1024 for next output

        config = GPTConfig(**config_args) # unpack eack config args
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd if not k.endswith('.attn.bias')] # remove buffer , in our model we can even set persistance to false so no need of this

        #initialize a Huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        #Match all the parameter names and shapes according to the transformer model
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bais')] #ignore them
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bais')] #ignore them
        transposed = ['attn.c_attn.weight','attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        #According to video series: Openai checkpoints use a "Conv1d" module, but we only want to use vanila
        #as a result we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"MISMATCHED KEYS: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                #Special word with openai's conv1d weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                #vanila copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model 
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        #Begin with all the candidate parameters that require gradients
        param_dict = {pn: p for pn,p in self.named_parameters()}
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}

        #Create Optim groups. Any parameters that is 2D will be weight decayed (must keep 1D), if not no....
        #ex: all weight tensors in matmul + embedding decay, all biases and layernorms don't
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]   # Decay all 2D params
        nodecay_params = [p for n,p in param_dict.items() if p.dim() <2]  #ex 1D params

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}]

        if master_process:
            num_decay_params =   sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"Num decayed param tensors: {len(decay_params)}, with {num_decay_params:,} paramters decayed")
            print(f"Num non-decayed param tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters kept")

        #Create AdamW optimizer and use the fused version (for fast computation)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.96), eps=1e-8, fused=True)

        return optimizer
#---------------------------------------------------------------------------------------------------------------------

def load_tokens(filename):
    np_arrays = np.load(filename)
    tensors = torch.tensor(np_arrays, dtype=torch.long)
    return tensors


#Proper loading data with correct batch dimensions and DDP
class DataLoader:
    def __init__(self,B,T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train','val'}

        #Shard Files (fineweb edu 10B) each consisting of 100M tokens
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)  #100 list 
        shards = [s for s in shards if split in s] #Breakdown the list
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0 , f"No shards were found for split {split}"
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")

        

        """
        #Tiny shakesphere dataset training consisting of 1 Million Tokens
        with open('input.txt', 'r') as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding('gpt2')
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B *T)} batches") """

        #Current state
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        #process 0 = 0
        #process 1 = B * T * 1
        #process 2 = B * T * 2  so on so forth.....

        #Batch chunks : |--rank0---|,  |--rank1---|, |--rank2---|, |--rank3---|
        #Ex: Current position(rank=0)  = 8 * 1024 * 0 (process_rank)    ---> 0
        #Sequentially batch allocating [0: (0+8*1024 +1) =  8193]       ---> [0:8193]

        #Ex: Current position(rank=1)  = 8 * 1024 * 1 (process_rank)       ---> 8192
        #Sequentially batch allocating [8192: (8192 + 8*1024 +1) =  16385] ---> [8192:16385]

        self.reset()  #to set the dataloader

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B,T = self.B, self. T #reminder
        buf = self.tokens[self.current_position  : self.current_position+ B*T +1] #Jump over the batch by batch +1 (for y)
        x = (buf[:-1]).view(B,T) # inputs
        y = (buf[1:]).view(B,T)  #targets

        self.current_position += B*T * self.num_processes
        #If we ran out of data just loop back around to 0
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            #Adding shards
            self.current_shard = (self.current_shard + 1 ) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        
        return  x,y
        
    



"""
#----------------------------------------------------------
#test without a dataloader

tokenizer = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()  #Input data

text = text[:1000]
tokens = tokenizer.encode(text)

B,T = 4,32  # 4 rows 32 (sentence columns) for the training
buf = torch.tensor(tokens[:B*T + 1])  # See jupyter notebook for more in here :128 + 1 tokens
buf = buf.to(device)
x = buf[:-1].view(B,T) #(0,128)
y = buf[1:].view(B,T) #(1,129)  Target Tensor"""





"""num_return_sequences = 5
max_length = 30


# Initialize tokenizer and prepare input
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
idx = tokens.to(device)  # Move input to same device as model

# Initialize model and move to device
model = GPT(GPTConfig())
model = model.to(device)
model.eval()


#Geneate output from the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while idx.size(1) < max_length:
    #forward the model to get the logits
    with torch.no_grad():
        logits = model(idx) # (B,T,vocab_size)
        #Take the logits for the very last position
        logits = logits[:, -1, :]  # (B,vocab_size)
        probabilities = F.softmax(logits, dim = 1)
        #Keep the top 50 probalities and clamp up all others 
        #Just like huggingface pipeline default in here (5,50)
        topk_probs, topk_indices = torch.topk(probabilities,50, dim=-1)
        #Select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs,1) # (B,1)
        #Gather the corresponding indices
        xcol = torch.gather(topk_indices,-1,ix) #(B,1)
        #append to the sequence
        idx = torch.cat((idx,xcol), dim = 1)

#print individual raw outputs
for i in range(num_return_sequences):
    tokens = idx[i, :max_length].tolist()
    decode = enc.decode(tokens)
    print(">", decode)     """

#Train with DDP 
#-----------------------------------------------------------------------------
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
#torchrun command sets the env variable RANK,LOCAL_RANK, and WORLD_SIZE

#Normal Launch;
    #python train.py
    #DDP launch for 4 gpus
    #torchrun --standalone --nproc_per_node=world_size train.py




ddp = int(os.environ.get('RANK', -1)) != -1 #Is this a ddp run check?

if ddp:
    #Set device appropriately according to the rank we set
    assert torch.cuda.is_available(), "You cant continue ahead without CUDA for ddp"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  #used in multi node training
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # Number of Nodes (GPU Servers)
    device = f'cuda:{ddp_local_rank}' # which GPU to use if thre is more than one GPU
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # This will do loading and saving ckpts, loggings , etc..

else:
    #non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    #Auto detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#---------------------------------------------------------------

#Gradient Accumulation (See point 5. in notes.txt)
total_batch_size = 491520  #  2 **29, 0.5M in number of tokens
B = 4 # Micro Batch size (how many rows processing in a single backward) 16*1024 =16,384 per forward backward
T = 1024 # Sequence length (num_words)
assert total_batch_size % (B*T*ddp_world_size) == 0, "Total batch size must divide by B * T * DDP_world_size"
accumulation_step = total_batch_size // (B *T * ddp_world_size)  # 32 in our case , 3 or 4 node (in GPU Server) 

if master_process:  #  we dont want to print every time each local running loop so we only print in master
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation step: {accumulation_step}") #(524288/(16*1024*4))

    print(f"Effective batch size per GPU: {B * T}")
    print(f"Total effective batch size: {B * T * ddp_world_size * accumulation_step}")


#Check GPU usage status by stopping from here
#print("This is GPU: ", ddp_rank)
#print("bye")
#import sys; sys.exit(0)





#Nomal Data Loder
#train_loader = DataLoader(B = B, T = T)  # Again: B number of lines (rows), T number of sentences in that row
                                            # B = 0.5e6/1024 = 488 ; GPT-2 0.5M Batch size


#Partitioned Data loader for DDP
train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank,num_processes=ddp_world_size,split="val")



#Tensorfloat32 initialization for fast computation
#Every time we see a matrix multiplication in our model utilizes the TF32 precision (see more) (8x times faster)
#instead if 19.5 flops 156 flops : ) Happy Happy Happyyyy.....
torch.set_float32_matmul_precision('high') #try training without this so that we can compare computation time

#create the model
model = GPT(GPTConfig()) #Because of seed the same model duplicated to each GPU
model.to(device)
#model = torch.compile(model)  # again for faster computation (pytorch.org/tutorials/introduction to torch.compile)
#Simply: instead of running the code line by line torch looks at the whole code and compile it without intepreter
#And Store memory inside of chips(cores) instead of each exchange through HBM (High Brandwith Memory) which is like RAM in CPU (HBM in GPU)

#Wrap the model to DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])  #See line 4 in notes to get a good idea

#get out previous modules along the wrapped DDP model
raw_model = model.module if ddp else model # raw unwrapped model if no GPU



#lr scheduling
max_lr        =  6e-4
min_lr        =  max_lr * 0.1
warmup_steps  =  715


def get_lr(step):
    #1. Linear warmup for warmup_iters steps
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps 

    #2. if step> lr_decay_iters, return min learning rate
    if step > GPTConfig.epochs:
        return min_lr

    #3. in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (GPTConfig.epochs - warmup_steps) 
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  #coeffiecient begins at 1 and gets to 0
    return min_lr + coeff * (max_lr - min_lr) 



def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm






import time
#Loop out the loss and optimize : )
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8)  #Hyperparam Betas: in GPT-3
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)
#----------------------------------------------------------------------
#Create a Log
log_dir = "log"
os.makedirs(log_dir, exist_ok = True)
log_file = os.path.join(log_dir, f"log.txt")
with open (log_file, "w") as f: #Write logs inside this
    if master_process:
        print("Log Initialized Successfully.......")
    pass #This starts empy so later we are going to append to it
    





#----------------------------------------------------------------------

if master_process:
    print("Train Begins...........") 

for epoch in range(GPTConfig.epochs):
    t0 = time.time()
    last_epoch = (epoch == GPTConfig.epochs -1)

    #Calculate the Val Loss along the way
    #Once in a while evaluate our val loss
    if epoch % 250 == 0 or last_epoch:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation Loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{epoch} val {val_loss_accum.item():.4f}\n")
            if epoch > 0 and (epoch % 1000 == 0 or last_epoch):
                #Save the model every 1000 epochs
                ckpt_path = os.path.join(log_dir,f"Epoch{epoch:05d}_ckpt.pt")
                ckpt = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': epoch,
                    'val_loss': val_loss_accum.item(),
                    'optimizer': optimizer.state_dict(),
                    }
                torch.save(raw_model.state_dict(), ckpt_path)
                print(f"Saved model checkpoint to {ckpt_path}")


    
    from hellaswag import iterate_examples, render_example
    #Ongoing Evaluation with HellaSweg
    if (epoch % 200 == 0 or last_epoch):
        num_correct_norm = 0
        num_total = 0

        for i,example in enumerate(iterate_examples("val")):
            #To engage all the processes
            if i % ddp_world_size != ddp_rank:
                continue

            #Render the examples into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            #Get the logits outputs
            with torch.no_grad():
                with torch.autocast(device_type= device, dtype=torch.bfloat16):
                    logits,loss = model(tokens)
                pred_norm = get_most_likely_row(tokens,mask,logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        #Reduce the stats across all the processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype= torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)    
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)        
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm/num_total

        if master_process:
            print(f"Hellaswag accuracy: {num_correct_norm} / {num_total} = {acc_norm:.4f}")
            with open (log_file, "a") as f:
                f.write(f" {epoch} Hellaswag {acc_norm:.4f}\n") 


    #Ongoing Evaluation to see if the model works fine
    if epoch > 0 and epoch % 50 == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokenizer = tiktoken.get_encoding('gpt2')
        tokens = tokenizer.encode("Hello this is AI model,")
        tokens = torch.tensor(tokens, dtype=torch.long)  #Need to convert to tensors
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(31 + ddp_rank)

        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = raw_model(xgen)
                logits = logits[:, -1, :] # (B,Vocab_size)
                probs = F.softmax(logits, dim=-1) #Probabillities
                #TopK sampling to get the top 50 logits
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator= sample_rng)
                xcol = torch.gather(topk_indices, -1 , ix)
                xgen= torch.cat((xgen,xcol), dim=1)
        
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = tokenizer.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    #Begin train the model
    raw_model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    #Gradient Accumalation for large batch size training
    for micro_step in range(accumulation_step):
        x,y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)

        #bfloat16 for fast computation than default float16 Used only for logits calculation which is already wrapped in the model
        with torch.autocast(device_type=device, dtype=torch.bfloat16): #Note that this is only possible at ampere
            logits, loss = raw_model(x,y) 
            #import code; code.interact(local=locals()) #We can use this to interrupt the process and code in terminal
            #Once only logits =bfloat16 but others like wte,pos are still float32 (this is called mixed precision)
            #More info: pytorch.org/Automatic Mixed Precision package/torch.amp/CUDA Ops that can autocast to float16
        lossn = (loss /accumulation_step)  # see 6 row at Jupyternotebook (play.ipynb)
        loss_accum += loss.detach()/  accumulation_step

        #Not to average out gradients in every single microstep just need only the very last accumulated gradient
        if ddp:
            model.require_backward_grad_sync = (micro_step == accumulation_step - 1)
            #syncronize only the very last step

        lossn.backward()
    
    #Average on every single loss accumulated
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    #So this loss_accum exists on all rank and when we call reduceop it well create the average
    #across all and deposits average on all the ranks 


    #Gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # max_norm  see line 22(notes.txt)

    #Cosine Decay Learning Rate Scheduling
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:  #This is how we set lr for individual epochs before updating weights
        param_group['lr'] = lr
    
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)  #Time difference for each epoch calculation
    processed_tokens = train_loader.B * train_loader.T * accumulation_step * ddp_world_size
    tokens_per_sec = processed_tokens / dt

    if master_process:  #only print in 0th rank's iter
        print(f"Epoch: {epoch} | Loss: {loss_accum.item():.5f} | lr: {lr:.4e} | Norm: {norm:.4f} | Time: {dt*1000:.2f}ms | Tok/sec: {tokens_per_sec: .2f}")

if master_process:
    torch.save(raw_model.state_dict(), "/Checkpoints/ckpts1.pt") #save the model every epoch

if ddp:
    destroy_process_group()




#Evaluate by taking the outputs






