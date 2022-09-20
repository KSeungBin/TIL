import torch
import torch.nn as nn


class Block(nn.Module): # nn.Module을 상속받은 class는 init, forward 함수 2개를 override 해줘야 함
    # init : 필요한 기능 정의
    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4): # bath_normalization = False인 경우 Dropout ON
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        super().__init__()
        
        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        self.block = nn.Sequential(  # Block class는 1 layer로 이루어져 있음
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size),
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)  # y_hat = f_\theta(x)
        # |y| = (batch_size, output_size) # y는 long tensor(one-hot index로 이루어진 vector), y_hat은 one-hot vector(tabular data or matrix 형태)
        
        return y

    
class ImageClassifier(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=[500, 400, 300, 200, 100], # hidden size도 결국엔 n_layers configuration만 입력하면 등차수열로 자동으로 결정되게 코딩할 것
                 use_batch_norm=True,
                 dropout_p=.3):
        
        super().__init__()

        assert len(hidden_sizes) > 0, "You need to specify hidden layers"

        last_hidden_size = input_size
        blocks = []
        for hidden_size in hidden_sizes:
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size
        
        self.layers = nn.Sequential(
            *blocks,  # n_layers 5 : 4개의 block + 1 Linear layer => input size와 output size를 5등분하는 등차수열 자동으로 결정됨
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1), # y_hat이 특정 class에 대한 log 확률 분포값으로 변환됨 -> NLL Loss 사용해야 함
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)        
        y = self.layers(x)
        # |y| = (batch_size, output_size)
        
        return y
