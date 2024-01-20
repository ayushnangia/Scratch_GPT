from my_tokenize import xb,yb,get_batch
from examin import decode,vocabs as vocab_size 
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)
    def forward(self,idx, targets=None):
        logits=self.token_embedding_table(idx)
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,_=self(idx)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx

n=BigramLanguageModel(vocab_size)
out,loss=n(xb,yb)
print(out.shape)
print(loss)
optimizer=torch.optim.AdamW(n.parameters(),lr=1e-3)
batch_size=32
for s in range(10000):
    xb,yb=get_batch('train')
    logits,loss=n(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

print(decode(n.generate(idx=torch.zeros((1,1),dtype=torch.long),max_new_tokens=200)[0].tolist()))