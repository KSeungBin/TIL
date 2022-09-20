# watch -n .25 nvidia-smi
# top
# python train.py --help
# python train.py --model_fn model.pth --gpu_id 0 --batch_size 512 --n_epochs 20 --model rnn --rnn_hidden_size 128 --rnn_dropout_p .1


import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from mnist_classification.data_loader import get_loaders
from mnist_classification.trainer import Trainer

from mnist_classification.models.fc_model import FullyConnectedClassifier
from mnist_classification.models.cnn_model import ConvolutionalClassifier
from mnist_classification.models.rnn_model import SequenceClassifier

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--model', type=str, default='fc')

    # rnn을 위한 argument 추가 
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--dropout_p', type=float, default=.2)

    # gradient clipping을 위한 threshold 설정 : gradient descent의 방향은 유지한 채 이동거리만 줄이는 것
    # rnn 이외의 architecture를 사용하더라도, gradient가 불안정해 loss가 튀는 현상이 발생하면 gradient clipping 고려하기
    p.add_argument('--max_grad', type=float, default=-1) # default를 -1로 두어, max_grad가 0보다 큰 경우에만 gradient clipping 적용

    config = p.parse_args()

    return config


def get_model(config):
    if config.model == 'fc':
        model = FullyConnectedClassifier(28**2, 10)
    elif config.model == 'cnn':
        model = ConvolutionalClassifier(10)
    elif config.model == 'rnn':
        model = SequenceClassifier(
            input_size = 28,
            hidden_size = config.hidden_size,
            output_size = 10,
            n_layers = config.n_layers,
            dropout_p = config.dropout_p,
        )
    else:
        raise NotImplementedError('You need to specify model name.')

    return model


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    train_loader, valid_loader, test_loader = get_loaders(config)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    model = get_model(config).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)

if __name__ == '__main__':
    config = define_argparser()
    main(config)