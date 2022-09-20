import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):

    # 1 block을 통과하면 : image size는 h와 w가 각각 절반씩 줄어 면적은 1/4씩 줄어들고, 동시에 채널수는 2배 증가하도록 구현
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1), # 3*3 kernel + 1 padding = 입출력 size 동일
            nn.ReLU(),
            nn.BatchNorm2d(out_channels), # FC Layer에서 BatchNorm1d 사용할 떄는 출력의 크기를 인자로 넣었는데, 2d에서는 output channel의 개수를 넣어줌
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1), # 2 stride로 출력 size(h,w)를 절반씩 줄임
            nn.ReLU(),
            nn.BatchNorm2d(out_channels), # output channel의 개수는 첫번째 conv를 통과할 때와 동일하게 구성
        )

    def forward(self, x):
        # grayscale : in_channels = 1   /  color : in_channels = 3
        # |x| = (batch_size, in_channels, h, w)

        y = self.layers(x)
        # |y| = (batch_size, out_channels, h, w)

        return y


class ConvolutionalClassifier(nn.Module):

    def __init__(self, output_size):  # input image size는 (28,28)로 고정해야 별도의 layer를 구성하지 않아도 됨. 따라서, output_size만 hyper param으로 받는다
        self.output_size = output_size

        super().__init__()

        self.blocks = nn.Sequential( # |x| = (bs, 1, 28, 28)
            ConvolutionBlock(1, 32), # (bs, 32, 14, 14)
            ConvolutionBlock(32, 64), # (bs, 64, 7, 7)
            ConvolutionBlock(64, 128), # (bs, 128, 4, 4)
            ConvolutionBlock(128, 256), # (bs, 256, 2, 2)
            ConvolutionBlock(256, 512), # (bs, 512, 1, 1) = (bs, 512) = 512차원의 vector가 batch_size만큼 있는 것과 동일(image에 대한 512개의 feature vector)
        )
        self.layers = nn.Sequential(
            nn.Linear(512, 50), # (bs,512) -> (bs, 50)
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size), # (bs, 50) -> (bs, 10)
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        assert x.dim() > 2  # 실수로 flatten하지 않도록 assert 걸어둠 : (bs, 784) size의 input이 들어오면 error

        if x.dim() == 3:
            # |x| = (batch_size, h, w)
            x = x.view(-1, 1, x.size(-2), x.size(-1))   # (bs, 28, 28) -> (bs, # of channels, H, W) : CNN의 INPUT TENSOR 형태는 (N,C,H,W)
        # |x| = (batch_size, 1, h, w)

        z = self.blocks(x)
        # |z| = (batch_size, 512, 1, 1)

        y = self.layers(z.squeeze())   # squeeze하면 1을 없애 (bs, 512) size로 변환됨
        # |y| = (batch_size, output_size) = (bs, 10)

        return y
