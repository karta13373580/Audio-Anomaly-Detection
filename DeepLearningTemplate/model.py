# import
from pytorch_lightning import LightningModule
import sys
from os.path import dirname, basename
import torch.nn as nn
import torch
import torch.optim as optim


# def
def load_from_checkpoint(device, checkpoint_path, model):
    device = device if device == 'cuda' and torch.cuda.is_available(
    ) else 'cpu'
    map_location = torch.device(device=device)
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def load_from_checkpoint_for_supervised_model(
    device,
    checkpoint_path,
    classes,
    model,
):
    device = device if device == 'cuda' and torch.cuda.is_available(
    ) else 'cpu'
    map_location = torch.device(device=device)
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    # change the number of output
    for key in checkpoint['state_dict'].keys():
        if 'classifier.bias' in key or 'classifier.weight' in key:
            if checkpoint['state_dict'][key].shape[0] != len(classes):
                temp = checkpoint['state_dict'][key]
                checkpoint['state_dict'][key] = torch.stack([temp.mean(0)] *
                                                            len(classes), 0)
    # change the weight of loss function
    if model.loss_function.weight is None:
        if 'loss_function.weight' in checkpoint['state_dict']:
            # delete loss_function.weight in the checkpoint
            del checkpoint['state_dict']['loss_function.weight']
    else:
        # override loss_function.weight with model.loss_function.weight
        checkpoint['state_dict'][
            'loss_function.weight'] = model.loss_function.weight
    model.load_state_dict(checkpoint['state_dict'])
    return model


# class
class BaseModel(LightningModule):
    def __init__(self, optimizers_config, lr, lr_schedulers_config) -> None:
        super().__init__()
        self.optimizers_config = optimizers_config
        self.lr = lr
        self.lr_schedulers_config = lr_schedulers_config

    def import_class_from_file(self, filepath):
        sys.path.append('{}'.format(dirname(filepath)))
        filename = basename(filepath)[:-3]
        # assume the class name in file is SelfDefinedModel
        class_name = 'SelfDefinedModel'
        # check if the class_name exists in the file
        exec('import {}'.format(filename))
        classes_in_file = dir(eval('{}'.format(filename)))
        assert class_name in classes_in_file, 'please check the self defined model.\nthe {} does not exist the {} class.'.format(
            filepath, class_name)
        exec('from {} import {}'.format(filename, class_name))
        return eval(class_name)

    def create_loss_function(self, loss_function_name):
        assert loss_function_name in dir(
            nn
        ), 'please check the loss_function_name argument.\nloss_function: {}\nvalid: {}'.format(
            loss_function_name, [v for v in dir(nn) if v[0].isupper()])
        return eval('nn.{}()'.format(loss_function_name))

    def parse_optimizers(self, params):
        optimizers = []
        for optimizer_name in self.optimizers_config.keys():
            if optimizer_name in dir(optim):
                if self.optimizers_config[optimizer_name] is None:
                    # create an optimizer with default values
                    optimizers.append(
                        eval('optim.{}(params=params, lr={})'.format(
                            optimizer_name, self.lr)))
                else:
                    # create an optimizer using the values given by the user
                    optimizer_arguments = [
                        '{}={}'.format(a, b) for a, b in
                        self.optimizers_config[optimizer_name].items()
                    ]
                    optimizer_arguments = ','.join(optimizer_arguments)
                    optimizers.append(
                        eval('optim.{}(params=params, lr={}, {})'.format(
                            optimizer_name, self.lr, optimizer_arguments)))
            else:
                assert False, 'please check the optimizer name in the optimizers_config argument.\noptimizer name: {}\nvalid: {}'.format(
                    optimizer_name, [v for v in dir(optim) if v[0].isupper()])
        return optimizers

    def parse_lr_schedulers(self, optimizers):
        lr_schedulers = []
        for idx, lr_scheduler_name in enumerate(
                self.lr_schedulers_config.keys()):
            if lr_scheduler_name in dir(optim.lr_scheduler):
                if self.lr_schedulers_config[lr_scheduler_name] is None:
                    # create a learning rate scheduler with default values
                    lr_schedulers.append(
                        eval(
                            'optim.lr_scheduler.{}(optimizer=optimizers[idx])'.
                            format(lr_scheduler_name)))
                else:
                    # create a learning rate scheduler using the values given by the user
                    lr_schedulers_arguments = [
                        '{}={}'.format(a, b) for a, b in
                        self.lr_schedulers_config[lr_scheduler_name].items()
                    ]
                    lr_schedulers_arguments = ','.join(lr_schedulers_arguments)
                    lr_schedulers.append(
                        eval(
                            'optim.lr_scheduler.{}(optimizer=optimizers[idx], {})'
                            .format(lr_scheduler_name,
                                    lr_schedulers_arguments)))
            else:
                assert False, 'please check the learning scheduler name in the lr_schedulers_config argument.\nlearning scheduler name: {}\nvalid: {}'.format(
                    lr_scheduler_name,
                    [v for v in dir(optim) if v[0].isupper()])
        return lr_schedulers

    def configure_optimizers(self):
        optimizers = self.parse_optimizers(params=self.parameters())
        if self.lr_schedulers_config is not None:
            lr_schedulers = self.parse_lr_schedulers(optimizers=optimizers)
            return optimizers, lr_schedulers
        else:
            return optimizers
