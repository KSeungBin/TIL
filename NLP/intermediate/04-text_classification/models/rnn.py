import torch.nn as nn


class RNNClassifier(nn.Module):

    # 필요한 변수 저장, forward에서 호출한 layers를 미리 선언
    def __init__(
        self,
        input_size,    # number of vocabulary: 사용하는 copus를 torchtext가 자동으로 읽어와서 vocab dict를 만들어주니, 그 size를 넣으면 된다. (corpus dependent hyper parameter)
        word_vec_size, # word embedding vector가 몇 차원으로 projection 될 것인가
        hidden_size,   # bi-directional LSTM의 hidden size: hidden state와 cell state의 size
        n_classes,
        n_layers=4,    # bi-directional LSTM의 layer 개수
        dropout_p=.3,  # LSTM 내부의 layers 간 dropout
    ):
        self.input_size = input_size  # vocabulary_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        # embedding layer: linear layer와 수학적으로 동일
        self.emb = nn.Embedding(input_size, word_vec_size) # 입력: vocab size / 출력: 지정한 차원의 word embedding vector
        
        # layer 선언
        self.rnn = nn.LSTM(
            input_size=word_vec_size, # embedding layer의 결과값을 입력으로 받는다
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,         # bi-directional RNN의 default: (length, bs, hs*2) -> batch_first를 True로 하면: (bs, length, hs*2)
            bidirectional=True,       # non-autoregressive task로 한번에 입력이 들어오기 때문에 bi-directional을 당연히 True로 놓기
        )

        # generator: Softmax를 통과시키기 전에 차원 축소
        # bi-directional LSTM의 경우 정방향&역방향 2개의 hidden state가 나오기 때문에 *2를 해서 받아주고, number of classes(=2)로 차원 축소해준다 
        # binary classification의 정석은 hidden state를 하나로 빼주고, sigmoid를 씌워 BCE Loss를 넣어줘야 하지만 genralization을 위해 아래와 같이 코드 작성
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        # LogSoftmax의 속도가 조금 더 빠르다
        self.activation = nn.LogSoftmax(dim=-1) # n_classes 차원에 softmax를 씌워 확룰값 구한다


    # problem definition: 문장을 입력받아(mini-batch), 각 mini-batch의 samples별 class별 확률값을 return
    def forward(self, x):
        # |x| = (bs, length, |V|) -> (bs, length, 1) = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x) # 출력의 첫번쨰는 output, 두번쨰는 마지막 time step의 hidden state와 cell state
        # |x| = (batch_size, length, hidden_size * 2)
        y = self.activation(self.generator(x[:, -1])) # 마지막 time step의 output만 slicing: (bs, 1, hs*2)에서 index 값만 가져오므로 (bs, hs*2)
        # |y| = (batch_size, n_classes)

        return y
