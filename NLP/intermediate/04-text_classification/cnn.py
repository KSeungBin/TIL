import torch
import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        n_classes,
        use_batch_norm=False,
        dropout_p=.5,

        # CNN Text Classifier에 특화된 hyper parameter
        window_sizes=[3, 4, 5],    # 3단어/4단어/5단어짜리 text classifier
        n_filters=[100, 100, 100], # 3단어짜리가 100개의 패턴(커널 개수)
    ):
        self.input_size = input_size  # vocabulary size
        self.word_vec_size = word_vec_size
        self.n_classes = n_classes
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        # window_size means that how many words a pattern covers.
        self.window_sizes = window_sizes
        # n_filters means that how many patterns to cover.
        self.n_filters = n_filters

        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_size) # embedding layer는 linear layer와 구현만 다를 뿐 수학적으로 동일
        # Use nn.ModuleList to register each sub-modules.
        self.feature_extractors = nn.ModuleList()  # CNN Classifier가 layer로 인식해 최적화 대상에 포함하려면 그냥 list가 아닌 ModuleList 객체를 사용해야 함
        for window_size, n_filter in zip(window_sizes, n_filters):
            self.feature_extractors.append(
                # nn.Sequential로 3단어/4단어/5단어짜리 sub-modules를 각각 만든다
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1, # We only use one embedding layer. 단어 당 하나의 word embedding vector만 들어오게 된다.
                        out_channels=n_filter,
                        kernel_size=(window_size, word_vec_size),
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(n_filter) if use_batch_norm else nn.Dropout(dropout_p),
                )
            )

        # An input of generator layer is max values from each filter.
        self.generator = nn.Linear(sum(n_filters), n_classes)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

    # input: one-hot mini-batch tensor of sentences
    # output: probability of each class
    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        min_length = max(self.window_sizes) # 5
        if min_length > x.size(1):
            # Because some input does not long enough for maximum length of window size,
            # we add zero tensor for padding.
            pad = x.new(x.size(0), min_length - x.size(1), self.word_vec_size).zero_() # 'self.word_vec_size' or 'x.size(-1)' 
            # |pad| = (batch_size, min_length - length, word_vec_size)
            x = torch.cat([x, pad], dim=1)  # length + (min_length - length) = min_length = 5
            # |x| = (batch_size, min_length, word_vec_size)

        # CNN - (BS, C, H, W) / TEXT - (BS, Time step, Dimension of one-hot vector(word embedding vector))
        # In ordinary case of vision task, you may have 3 channels on tensor,
        # but in this case, you would have just 1 channel,
        # which is added by 'unsqueeze' method in below:
        x = x.unsqueeze(1) # in_channels = 1로 선언했기 떄문에 unsqueeze(1)로 4차원으로 늘려주고 값은 1을 넣어준다.
        # |x| = (batch_size, 1, length, word_vec_size)

        cnn_outs = []
        for block in self.feature_extractors:
            cnn_out = block(x)
            # |cnn_out| = (batch_size, n_filter, length - window_size + 1, 1)

            # In case of max pooling, we does not know the pooling size,
            # because it depends on the length of the sentence.
            # Therefore, we use instant function using 'nn.functional' package.
            # This is the beauty of PyTorch. :)
            cnn_out = nn.functional.max_pool1d(
                input=cnn_out.squeeze(-1),     # (bs, n_filter, length - window_size + 1)
                kernel_size=cnn_out.size(-2)   # length - window_size + 1
            ).squeeze(-1)
            # |cnn_out| : (bs, n_filter, 1) -> (batch_size, n_filter)
            cnn_outs += [cnn_out]
        # Merge output tensors from each convolution layer.
        cnn_outs = torch.cat(cnn_outs, dim=-1)
        # |cnn_outs| = (batch_size, sum(n_filters)) = (bs, 300)
        y = self.activation(self.generator(cnn_outs))
        # |y| = (batch_size, n_classes) = (bs, 2)

        return y