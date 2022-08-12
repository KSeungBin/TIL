import torch

from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset):

    # flatten=True: 이미지를 벡터로 변환해 FC Layer(Module) 기반의 architecture를 위한 입력으로 만들어줌
    # flatten=False : CNN, RNN을 위한 입력
    def __init__(self, data, labels, flatten=True):
        self.data = data
        self.labels = labels
        self.flatten = flatten

        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]     # |x| = (28, 28)
        y = self.labels[idx]   # |y| = (1,)

        if self.flatten:       
            x = x.view(-1)     # |x| = (784,)

        return x, y


def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def get_loaders(config):
    x, y = load_mnist(is_train=True, flatten=False)

    # 10000장의 test set은 fix되어 있으니, train set만 정해진 비율로 train/valid로 split
    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0)) # 0 ~ 59999의 random한 수열을 생성
    train_x, valid_x = torch.index_select(
        x,
        dim=0,
        index=indices     # flatten=False이므로 |x|=(60000, 28, 28) -> 60000에 대해서 random한 수열에 대한 index select 수행
    ).split([train_cnt, valid_cnt], dim=0)  # 60000에 대해 split을 수행해 |train_x|=(48000,28,28), |train_y|=(12000,28,28)
    train_y, valid_y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=True),
        batch_size=config.batch_size, # batch size는 미리 지정해줘야 한다
        shuffle=True,                 # train_loader는 무조건 shuffling
    )
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_x, test_y = load_mnist(is_train=False, flatten=False)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader
