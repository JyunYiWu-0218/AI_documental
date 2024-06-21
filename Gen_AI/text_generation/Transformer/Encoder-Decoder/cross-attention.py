import torch.nn as nn
import torch
import math

punctuation="，。、[],.!"
temp_str = str() 
ids = {}
tokens = []

input_sentence_1 = input("輸入中文句子: ")
input_sentence_2 = input("輸入英文句子: ")

# 定義 cross-attention
"""
參數定義:
query: (input*dim_keyorquery)(因為 key = query) of vector
key: (input*dim_keyorquery)(因為 key = query) of vector
value: (input*dim_value) of vector
"""
class Cross_Attention(nn.Module):
    def __init__(self, dim_input, dim_KorQ, dim_V):
        super().__init__()
        self.dim_KorQ=dim_KorQ
        self.Weight_Q=nn.Parameter(torch.rand(dim_input, dim_KorQ))
        self.Weight_K=nn.Parameter(torch.rand(dim_input, dim_KorQ))
        self.Weight_V=nn.Parameter(torch.rand(dim_input, dim_V))
    
    def forward(self,input_1,input_2):
        input1_Q=input_1.matmul(self.Weight_Q)
        input2_K=input_2.matmul(self.Weight_K)
        input2_V=input_2.matmul(self.Weight_V)
        
        #根號運算
        #sqrt_k = math.sqrt(self.dim_KorQ)

        #Cross_Attention
        A=torch.softmax(input1_Q.matmul(input2_K.T), dim=-1)
        
        Cross_Attention=A.matmul(input2_V)
        return Cross_Attention

# 前處理輸入 (ids and token)
def Embedding_input(sentence:str, temp_str:str):
    for i in punctuation: 
        inputs = sentence.replace(i,'')

    for line in inputs: 
        temp_str+=line 

    for i,s in enumerate(str(temp_str)):
        sorts = {s:i}
        ids.update(sorts)

    for s in temp_str:  
        tokens.append(ids[s])
    

    input_tokens=torch.tensor(tokens)
    #100 tensors of size 80
    embedding=torch.nn.Embedding(100, 100)
    #產生embedding vector, detach() 防止反向傳播(Backpropagation)
    embedded_sentence=embedding(input_tokens).detach()
    return embedded_sentence

# 前處理輸入2 (ids and token)
def Embedding_input2(sentence:str, temp_str:str, dim_input:int):
    for i in punctuation: 
        inputs = sentence.replace(i,'')

    for line in inputs: 
        temp_str+=line 

    for i,s in enumerate(str(temp_str)):
        sorts = {s:i}
        ids.update(sorts)

    for s in temp_str:  
        tokens.append(ids[s])
    

    input_tokens=torch.tensor(tokens)
    #100 tensors of size 80
    embedding2=torch.nn.Embedding(70, dim_input)
    #產生embedding vector, detach() 防止反向傳播(Backpropagation)
    embedded_sentence=embedding2(input_tokens).detach()
    return embedded_sentence


#embedding
input1 = Embedding_input(sentence=input_sentence_1, temp_str=temp_str)
input2 = Embedding_input2(sentence=input_sentence_2, temp_str=temp_str, dim_input=input1.shape[1])

#參數
dim_input, dim_keyorquery, dim_value = input1.shape[1],32,48

#隨機種子,讓此文件由 rand() 輸出的資料都固定
#range: [-0x8000000000000000, 0xffffffffffffffff]
#[-9223372036854775808, 18446744073709551615]，超出該範圍將觸發RuntimeError報錯。
torch.manual_seed(123)

# CrossAttention
cross_attention=Cross_Attention(dim_input, dim_keyorquery, dim_value)

context_vectors=cross_attention(input1, input2)

print(context_vectors)
print(context_vectors.shape)


