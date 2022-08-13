from torchvision import models
from torch import nn


def set_parameter_requires_grad(model, freeze):
    for param in model.parameters():
        param.requires_grad = not freeze # freeze가 True이면, required_grad는 Flase가 되어 학습이 안 된다(=freeze)


def get_model(config):
    # You can also refer: 
    # - https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # - https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    model = None
    input_size = 0

    if config.model_name == "resnet":
        """ Resnet34
        """
        model = models.resnet34(pretrained=config.use_pretrained) # load pretrained weights (FALSE면 random init)
        set_parameter_requires_grad(model, config.freeze)

        # 1000개의 class가 아닌 2개의 class를 분류해야 하므로 FC Layers 단을 바꿀 것
        n_features = model.fc.in_features
        # FC Layer 이전 단(n_features)은 그대로 사용하고, n_classes만 우리가 classification할 class 개수로 바꿔줌(2개)
        model.fc = nn.Linear(n_features, config.n_classes)
        input_size = 224
    elif config.model_name == "alexnet":
        """ Alexnet
        """
        model = models.alexnet(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_features, config.n_classes) # VGG 역시 마지막 Layer를 바꿀 것(customize)
        input_size = 224
    elif config.model_name == "vgg":
        """ VGG16_bn
        """
        model = models.vgg16_bn(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_features, config.n_classes)
        input_size = 224
    elif config.model_name == "squeezenet":
        """ Squeezenet
        """
        model = models.squeezenet1_0(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        model.classifier[1] = nn.Conv2d(
            512,
            config.n_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        model.n_classes = config.n_classes
        input_size = 224
    elif config.model_name == "densenet":
        """ Densenet
        """
        model = models.densenet121(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier.in_features
        model.classifier = nn.Linear(n_features, config.n_classes)
        input_size = 224
    else:
        raise NotImplementedError('You need to specify model name.')

    return model, input_size
