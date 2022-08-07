import torch.nn as nn


class Autoencoder(nn.Module):
    # advanced code : n_layers를 세팅해서 layer 별 input, output value를 등차수열로 구성하는 것
    # 현재는 직관적인 이해를 위해 Sequential로 architecture를 구성함 
    
    def __init__(self, btl_size=2):
        self.btl_size = btl_size # bottle neck size가 10보다는 작아야 한다. 10보다 큰 값을 세팅하면, 다른 구간이 병목이 된다.
        
        super().__init__()
        
        # encoder와 decoder 구조가 반드시 대칭일 필요는 없다.
        # 비대칭 구조여도 되고, 둘 중 하나만 shallow 해도 된다.
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            # encoder, decoder의 마지막 layer는 activation function(non-linear function)이나 BN(regularizer) 없이 반드시 linear한 형태!
            nn.Linear(10, btl_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(btl_size, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 28 * 28),
        )

    # |x| = (batch_size, 28*28) = |y|
    # |z| = (batch_size, btl_size)   
    def forward(self, x):
        z = self.encoder(x)  # z는 입력 샘플(x)을 encoder에 통과시켜 출력된 값으로, 병목구간(hidden(latent) space)의 feature vector에 해당
        y = self.decoder(z)  # y는 z를 decoding하여 입력 이미지(x)와 가장 유사하게 복원된 출력 이미지
        
        return y