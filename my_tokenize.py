import torch 
from examin import encode
with open('input.txt','r',encoding='utf-8') as file:
    text = file.read()
data=torch.tensor(encode(text),dtype=torch.long)
#splitting data into train and validation
n=int(len(data)*0.8)
train_data=data[:n]
val_data=data[n:]


torch.manual_seed(42)
batch_size=4
block_size=8

def get_batch(split):
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1]for i in ix])
    return x,y
xb,yb=get_batch('train')

# for b in range(batch_size):
#     for j in range(block_size):
#         cont=xb[b:j+1]
#         targ=yb[b:j]
        
