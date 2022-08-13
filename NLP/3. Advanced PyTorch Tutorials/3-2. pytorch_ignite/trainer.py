# ignite를 활용해 event 기반의 프로그래밍을 하면 유지보수 편리하다.
# training하는 boiler template을 만들어놓으면 재활용하면서 딥러닝 연구 개발의 생산성을 높일 수 있다.
from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MyEngine(Engine):

    def __init__(self, func, model, crit, optimizer, config):
        # Ignite Engine does not have objects in below lines.
        # Thus, we assign class variables to access these object, during the procedure.
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func) # Ignite Engine only needs function to run.

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    # train : feed-forward -> loss 계산 -> back-propagation -> gradient descent -> 현재 상태 출력
    @staticmethod
    def train(engine, mini_batch): # data_loader.py의 MnistDataset class의 return 대로 mini_batch는 튜플(x,y)로 되어있음
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train() # Because we assign model as class variable, we can easily access to it.
        engine.optimizer.zero_grad()  # 한 번의 iteration마다 engine이 호출되므로 zero_grad 선언

        x, y = mini_batch
        x, y = x.to(engine.device), y.to(engine.device)

        # Take feed-forward
        # |x| = (bs, 784),   |y_hat| = (bs, 10) -> y_hat에는 각 차원(class)별 확률값이 들어있다 
        y_hat = engine.model(x)

        loss = engine.crit(y_hat, y) # scalar 값 
        loss.backward() # 모든 weights에 gradient 값이 채워짐

        # Calculate accuracy only if 'y' is LongTensor,
        # which means that 'y' is one-hot representation.
        # y가 long tensor이면 classification, y가 float tensor면 regression task
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))  # 현재 mini-batch에서의 accuracy
        else:
            accuracy = 0

        # parameter의 L2 Norm : 학습이 진행될수록 모델의 복잡도가 높아지면서 p_norm은 점점 커져야한다 = model parameter가 update되고 있다.
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        # gradient의 L2 Norm : gradient의 크기로 현재 loss surface가 얼마나 가파른지 판단할 수 있다.(학습의 안정성을 판단할 수 있는 지표)
        # 학습을 처음 시작하면 배울게 많아(오답이 많아) 기울기가 가파르다. 학습이 진행됨에 따라 g_norm이 줄어들면서 일정 숫자로 수렴해야 한다
        # SGD로 항상 모집단과 다르기 때문에 0으로 수렴하지는 않지만, 일정 숫자로 수렴하지 않고 g_norm 수치가 날뛰면 loss가 NaN이 되면서 학습이 제대로 안될 것
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # Take a step of gradient descent.
        engine.optimizer.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }


    # validate : feed-forward -> loss 계산 -> 현재 상태 출력
    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch
            x, y = x.to(engine.device), y.to(engine.device)

            y_hat = engine.model(x)

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }

    # attach : train과 validation의 현재 상황 출력(loss가 떨어지고 있는지 등을 확인하기 위함)
    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name): # loss의 running average를 자동으로 계산 및 저장
            # 위 train function의 return 값(dict 형태)에서 필요한 metric_name을 받아오면 RunningAverage 객체가 만들어지는데
            # 그걸 engine에 metric_name(str)으로 붙인다
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name, # 따라서, 꼭 이름이 metric_name일 필요는 없다
            )

        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch.
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED) # event handler 등록 : epoch이 끝났을 때 원하는 metric을 출력
            def print_train_logs(engine):
                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))

        validation_metric_names = ['loss', 'accuracy']
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss: # If current epoch returns lower validation loss,
            engine.best_loss = loss  # Update lowest validation loss.
            engine.best_model = deepcopy(engine.model.state_dict()) # Update best model weights.

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model': engine.best_model,
                'config': config,
                **kwargs
            }, config.model_fn
        )


class Trainer():

    def __init__(self, config):
        self.config = config

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader
    ):
        train_engine = MyEngine(
            MyEngine.train,
            model, crit, optimizer, self.config
        )
        validation_engine = MyEngine(
            MyEngine.validate,
            model, crit, optimizer, self.config
        )

        MyEngine.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        # training이 1 epoch 끝나면, validation의 1 epoch이 시작된다
        # 따라서 training을 run 시키면, validation이 언제 호출될지 알려줘야 한다
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler( # train_engine의 1 epoch이 끝났을 때 run_validation이 시작된다
            Events.EPOCH_COMPLETED, # event
            run_validation, # function
            validation_engine, valid_loader, # arguments
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            MyEngine.check_best, # function
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            MyEngine.save_model, # function
            train_engine, self.config, # arguments
        )

        # 앞까지는 붙이는 작업
        # 이제부터 train_engine을 실제 실행(n_epoch run)
        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model
