import torchvision
from torch import nn

NUM_CLASSES = 2


class Model:
    ...


# MODELS: dict[str, type[Model]] = {}


class Model:
    MODELS: dict[str, type[Model]] = {}

    @staticmethod
    def get(**kwargs):
        ...

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group('Model')
        parser.add_argument("--classes", type=int, default=NUM_CLASSES, help="Number of classes")
        return parent_parser

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Model.MODELS[cls.__name__] = cls


class ResNet18(Model):
    @staticmethod
    def get(classes, **kwargs):
        return torchvision.models.resnet18(num_classes=classes)

    # @staticmethod
    # def add_argparse_args(parent_parser):
    #     return parent_parser


class ResNet18_1layer_fc(Model):
    @staticmethod
    def get(classes, layer1, **kwargs):
        model = torchvision.models.resnet18(num_classes=layer1)
        model = nn.Sequential(model, nn.ReLU(), nn.Linear(model.fc.out_features, classes))
        return model

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("LitDataModule")
        parser.add_argument("--classes", type=int, default=NUM_CLASSES, help="Number of classes")
        parser.add_argument("--layer1", type=int, default=1000)
        return parent_parser
class ResNet18_1layer_fc_pretrained(Model):
    @staticmethod
    def get(classes, **kwargs):
        model = torchvision.models.resnet18(pretrained=True)
        model = nn.Sequential(model, nn.ReLU(), nn.Linear(model.fc.out_features, classes))
        return model

    # @staticmethod
    # def add_argparse_args(parent_parser):
    #     parser = parent_parser.add_argument_group("LitDataModule")
    #
    #     return parent_parser
class ResNet50(Model):
    @staticmethod
    def get(classes, **kwargs):
        return torchvision.models.resnet50(num_classes=classes)

    # @staticmethod
    # def add_argparse_args(parent_parser):
    #     return parent_parser

class efficientnet_b0(Model):
    @staticmethod
    def get(classes, **kwargs):
        return torchvision.models.efficientnet_b0(num_classes=classes)

    # @staticmethod
    # def add_argparse_args(parent_parser):
    #     return parent_parser
class densenet121(Model):
    @staticmethod
    def get(classes, **kwargs):
        return torchvision.models.densenet121(num_classes=classes)

    # @staticmethod
    # def add_argparse_args(parent_parser):
    #
    #     return parent_parser

class regnet_x_3_2gf(Model):
    @staticmethod
    def get(classes, **kwargs):
        return torchvision.models.regnet_x_3_2gf(num_classes=classes)

    # @staticmethod
    # def add_argparse_args(parent_parser):
    #     super().add_argparse_args(parent_parser)
    #     return parent_parser


def configure_argparser(parent_parser, model_name):
    return Model.MODELS[model_name].add_argparse_args(parent_parser)


def get_model(model_name, **kwargs) -> nn.Module:
    return Model.MODELS[model_name].get(**kwargs)
