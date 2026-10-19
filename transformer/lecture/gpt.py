#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
# <<<peace hard>>>
batch_size = 64       
block_size = 256       # 每条序列最长 256 个 token
max_iters = 5000       # 总共训练 5000 步
eval_interval = 500    # 每 500 步评估一次
learning_rate = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 200  # 每次评估采样200个batch估算loss
n_embd = 384  # 嵌入维度 / 隐藏层宽度  每个token的维度
n_head = 6 # 多头注意力头数
n_layer = 6 # Transformer 块层数
dropout = 0.2 # Dropout 概率

torch.manual_seed(1337)

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))  # 利用集合 
vocab_size = len(chars)
# i is number
stoi = {ch : i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s:[stoi[c] for c in s]
decode = lambda l:''.join([itos[i] for i in l])


data  = torch.tensor(encode(text),dtype=torch.long)
n  = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) -  block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
        pass

class Head(nn.module):
    # the diffrent of noe head attention and multi head attention:
    def __int__(self,head_size):
        super.__init__()
        # W_q * E = Q E为词向量空间  head_size = n_embd // n_head
        # nn.Linear(n_embd, head_size) 只变换输入张量的最后一维，前面的维度保持不变。  
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))  #block_size 每条序列最长是256个token
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape  # batch_size  time_step(每条样本有多少个token) c=channels，即 n_embd，嵌入维度
        k = self.key(x) # size = (B,T,hs)
        q = self.query(x)
        # 计算注意力分数
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T)T -> (B, T, T)  # 自注意力就来自这里
        wei.masked_fill(self.tril[:T,:T] ==  0,float('-inf'))
        wei = F.softmax(wei,dim= -1)
        # i cant see i cant see
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super.__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) #?不就是一个全连接层吗
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1).cat([])  # -1 是沿最后一维拼接
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):  # 前馈层 也就是多层感知机
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(n_embd*4,n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)
class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__() 
        head_size = n_embd // n_head       
        self.sa = MultiHeadAttention(n_head, head_size)  # self attention
        self.ffwd = FeedFoward(n_embd)  # feed foward 前馈神经网络
        self.ln1 = nn.LayerNorm(n_embd)  # 归一化层
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        # 一下是transformer paper method

        # x = self.ln1(x + self.sa(x))
        # x = self.ln2(x + self.ffwd(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table  = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.mormal_(module.weight,mean = 0.0,std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=0.2)
    def forward(self,idx,targets = None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B ,T ,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits,loss
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,block_size:]
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6,'M parameters')
optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate)

for iter in range(max_iters):
    if iter%eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
'''
模型的输入 idx 形状是 (B, T)
第一个维度 1 batch size 为 1 表示一次只生成一条文本；
第二个维度 1 序列长度为 1 表示初始只有一个字符 起始 token。
'''
context = torch.zeros((1,1),dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))



        


    
    

        













#%%

