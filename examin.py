with open('input.txt','r',encoding='utf-8') as file:
    text = file.read()
# print("lenght of dataset",len(text))

# print("first 100 characters of dataset\n",text[:100])

chars=sorted(list(set(text)))
vocabs=len(chars)
# print("".join(chars))
# print("total possible characters",vocabs)


## lets make a tokenizer for our dataset
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i , ch in enumerate(chars)}
encode=lambda s : [stoi[c] for c in s ]
decode=lambda l:"".join([itos[i] for i in l])

# print("encoded form of 'hello world'",encode('hello world'))
# print("decoded form of 'hello world'",decode(encode('hello world')))


