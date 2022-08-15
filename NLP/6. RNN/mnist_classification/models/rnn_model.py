import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):
    # FC Module : 784차원 입력/ CNN : (28, 28) 입력 / RNN : 1 row씩 입력을 받기 때문에 입력 size 28, 출력 size 28(-> 즉, 28차원의 row가 28개 들어온다)
    def __init__(
        self,
        input_size,
        hidden_size,  # default 64
        output_size,
        n_layers=4,   # LSTM은 depth에 대한 gradient vanishing을 해결하지 못하므로 최대 4까지 추천
        dropout_p=.2,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        # PyTorch 공식 document에서 LSTM 검색하고 Parameters 확인
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,    # batch_first의 default는 False (True로 해야 (bs, time_step=28, vector_size=28)형태로 입력할 수 있음)
            dropout=dropout_p,   # RNN은 Batch Normalization 기법을 사용할 수 없다 -> dropout 또는 layer normalization 기법을 사용할 수 있다
            bidirectional=True,  # default는 False. non-autoregressive task(입력이 처음부터 끝까지 주어짐)이므로 bidirectional LSTM 사용
                                 # auto-regressive task가 아닌 이상 True로 사용할 수 있을 때는 True로 놓고 사용하기
        )
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2), # 정방향 + 역방향
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, h, w)  # grayscale : h=time_step, w=1 time step 당 입력으로 들어오는 vector size

        # batch_first=True이므로 Output shape은 (bs, seq_len=28, num_directions(2)*hidden_size)
        z, _ = self.rnn(x)   # hidden state와 cell state의 initial value를 생략하면 자동으로 0으로 두고 진행. 출력도 h_0과 c_0 생략
        # |z| = (batch_size, h, hidden_size * 2)
        z = z[:, -1]  # 첫 번쨰 차원(batch)은 다 가져오고, 두 번째 차원은 마지막 time step에 대한 hidden state만 가져와라
        # |z| = (batch_size, hidden_size * 2)
        y = self.layers(z)  # 출력 size는 CNN과 동일하므로 이후로는 FC Module과 interface가 동일
        # |y| = (batch_size, output_size)

        return y
