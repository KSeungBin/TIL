import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from simple_ntc.trainer import Trainer
from simple_ntc.data_loader import DataLoader

from simple_ntc.models.rnn import RNNClassifier
from simple_ntc.models.cnn import CNNClassifier


def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)  # name of model network file
    p.add_argument('--train_fn', required=True)  # train file -> train:valid = 8:2
    
    p.add_argument('--gpu_id', type=int, default=-1) # cpu: -1 / gpu: 0 ~
    p.add_argument('--verbose', type=int, default=2) # 0: 아무것도 출력 안 함 / 1: epoch이 끝날 떄마다 출력 / 2: iteration마다 정보 출력

    p.add_argument('--min_vocab_freq', type=int, default=5)  # 5번 이상 나오는 단어만 classifier가 학습
    p.add_argument('--max_vocab_size', type=int, default=999999)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--word_vec_size', type=int, default= 256) # word vector dimension
    p.add_argument('--dropout', type=float, default=.3)

    p.add_argument('--max_length', type=int, default=256) # 나머지는 clipping 
    
    # rnn은 전체 문장의 흐름(context)을 보고, cnn은 구/절/단어의 패턴이 있는지 여부를 본다 -> ensemble로 동시에 학습 가능
    p.add_argument('--rnn', action='store_true')
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=4)

    p.add_argument('--cnn', action='store_true')
    p.add_argument('--use_batch_norm', action='store_true')
    p.add_argument('--window_sizes', type=int, nargs='*', default=[3, 4, 5])
    p.add_argument('--n_filters', type=int, nargs='*', default=[100, 100, 100])

    config = p.parse_args()

    return config


def main(config):
    loaders = DataLoader(
        train_fn=config.train_fn,
        batch_size=config.batch_size,
        min_freq=config.min_vocab_freq,
        max_vocab=config.max_vocab_size,
        device=config.gpu_id
    )

    print(
        # number of sentences
        '|train| =', len(loaders.train_loader.dataset),
        '|valid| =', len(loaders.valid_loader.dataset),
    )
    
    vocab_size = len(loaders.text.vocab) # training set
    n_classes = len(loaders.label.vocab) # number of classes = 2
    print('|vocab| =', vocab_size, '|classes| =', n_classes)

    if config.rnn is False and config.cnn is False:
        raise Exception('You need to specify an architecture to train. (--rnn or --cnn)')

    if config.rnn:
        # Declare model and loss.
        model = RNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            hidden_size=config.hidden_size,
            n_classes=n_classes,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        )
        optimizer = optim.Adam(model.parameters()) # model의 weight parameters가 iterative하게 return되면 adam으로 넘겨준다
        crit = nn.NLLLoss() # log softmax -> use NLL instead of cross entropy
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        rnn_trainer = Trainer(config)
        # best model exists in rnn_model
        rnn_model = rnn_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader,
            loaders.valid_loader
        )
    if config.cnn:
        # Declare model and loss.
        model = CNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            n_classes=n_classes,
            use_batch_norm=config.use_batch_norm,
            dropout_p=config.dropout,
            window_sizes=config.window_sizes,
            n_filters=config.n_filters,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        cnn_trainer = Trainer(config)
        cnn_model = cnn_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader,
            loaders.valid_loader
        )

    torch.save({
        'rnn': rnn_model.state_dict() if config.rnn else None,
        'cnn': cnn_model.state_dict() if config.cnn else None,
        # classify: 추론 단계에서 모델을 객체로 만든 후 load state_dict -> 따라서, 모델이 똑같이 만들어져야 하므로 현재 configuration을 알아야 함 
        'config': config,
        'vocab': loaders.text.vocab, # mapping dictionary is needed: tokenized sentences -> one-hot vector(index)
        'classes': loaders.label.vocab, # 0: positive / 1: negative
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)