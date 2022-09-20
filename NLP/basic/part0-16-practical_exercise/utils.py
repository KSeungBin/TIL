import torch


def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255. # 0~1사이의 값으로 normalization
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1) # reshape : (60000,28,28) -> (60000,784)

    return x, y


def split_data(x, y, train_ratio=.8):
    train_cnt = int(x.size(0) * train_ratio) # 60000 * 0.8
    valid_cnt = x.size(0) - train_cnt        # 60000 * 0.2

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return x, y


def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers - 1):
        hidden_sizes += [current_size - step_size] # step_size = 등차 -> 리스트 안에 input(784)~output(10) 사이의 값이 n_layers만큼 등분되어 등차수열로 원소가 저장됨
        current_size = hidden_sizes[-1]

    return hidden_sizes
