import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer

from utils import load_mnist
from utils import split_data
from utils import get_hidden_sizes


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=.3)

    p.add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train=True, flatten=True) # CNN이 아니므로 이미지를 flatten 해야 한다
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)

    print("Train:", x[0].shape, y[0].shape) # default : train: valid = 8:2 ratio로 split
    print("Valid:", x[1].shape, y[1].shape)

    input_size = int(x[0].shape[-1])   # 784
    output_size = int(max(y[0])) + 1   # 10

    model = ImageClassifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size,
                                      output_size,
                                      config.n_layers),
        use_batch_norm=not config.use_dropout,
        dropout_p=config.dropout_p,
    ).to(device)
    optimizer = optim.Adam(model.parameters()) # adaptive LR + momentum도 동시에 학습해 학습속도까지 빠름
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    # 학습 준비 완료되면 trainer 객체를 생성
    trainer = Trainer(model, optimizer, crit)

    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    # Save best model weights.
    # tuple이 아닌 dict type으로 저장하는 것이 convention
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
