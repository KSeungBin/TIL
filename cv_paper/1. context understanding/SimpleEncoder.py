import torch

def multi_head_attention(Q, K, V):
    num_batch, num_head, num_token_length, att_dim = K.shape
    Q = Q / (att_dim**0.5) 
     
    attention_score = Q @ K.permute(0,1,3,2)

    attention_score = torch.softmax(attention_score, dim=3)

    Z = attention_score @ V 

    return Z, attention_score



class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V):
     # if Q = K = V, that is self-attention. Otherwise, cross-attention
     # token_length : batch마다 number of words가 다르기 때문에 보통은 max 값을 정한다. 32를 max로 둔 경우, n batch의 words 수가 8일 때 24개의 zero padding을 준다.
     num_batch, num_head, num_token_length, att_dim = K.shape
     Q = Q / (att_dim**0.5) # d_k 대신 attention dimension 사용
     
     # num_batch, num_head, num_token_length, num_token_length
     attention_score = Q @ K.permute(0,1,3,2) # 2차원인 경우 K.T -> 현재는 4차원이므로 permute

     attention_score = torch.softmax(attention_score, dim=3)

     # num_batch, num_head, num_token_length, att_dim
     Z = attention_score @ V 

     return Z, attention_score

# encoder에는 주로 self-attention이 들어온다
class Encoder(torch.nn.Module):
    def __init__(self, hidden_dim, num_head):
       super().__init__()

       self.num_head = num_head
       self.hidden_dim = hidden_dim # embedding dimension
       # hidden_dim을 num_head로 나누었을 때 나누어 떨어지지 않으면 오류를 발생시키는 assert문을 넣어주기도 함

       self.MHA = MultiHeadAttention()
       
       # Wx + b 에서 matrix multiplication이므로 bias는 생략
       self.W_Q = torch.nn.Linear(hidden_dim, hidden_dim, bias=False) # Matrix로 하면 broadcast 연산을 신경써야 하므로 Linear가 더 편하다
       self.W_K = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
       self.W_V = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

       self.W_O = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)


    def to_multihead(self, vector):
        num_batch, num_token_length, hidden_dim = vector.shape
        att_dim = hidden_dim // self.num_head   # embedding dimension // attention head
        vector = vector.view(num_batch, num_token_length, self.num_head, att_dim)
        vector = vector.permute(0,2,1,3)        # [num_batch, num_head, num_token_length, att_dim]
        return vector


    def forward(self, input_Q, input_K, input_V):
        # input_Q :[num_batch, num_token_length, hidden_dim]

        Q = self.W_Q(input_Q) # projection(linear 변환)해도 hidden to hidden이므로 Q는 변하지 않는다 : [num_batch, num_token_length, hidden_dim]
        K = self.W_K(input_K)
        V = self.W_V(input_V)

        # split : hidden_dim을 (num_head, att_dim)으로 쪼개기
        Q = self.to_multihead(Q)  # [num_batch, num_head, num_token_length, att_dim]
        K = self.to_multihead(K)
        V = self.to_multihead(V)

        Z, attention_score = self.MHA(Q,K,V)