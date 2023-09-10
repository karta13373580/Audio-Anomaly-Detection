# import
import argparse
from ruamel.yaml import safe_load
from os.path import isfile, isdir, abspath

# def


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        config = safe_load(f)
    assert not (config is None), 'the {} file is empty.'.format(filepath)
    return config


# class


class ProjectParameters:
    def __init__(self, ) -> None:
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.MetavarTypeHelpFormatter)
        self.parser.add_argument(
            '--config',
            type=str,
            required=True,
            help=
            'the project configuration path. if given None, it will not be loaded (it needs to be used with dont_check).'
        )
        self.parser.add_argument(
            '--dont_check',
            action='store_false',
            help=
            'whether to check kwargs, if given, kwargs will not be checked.')
        self.parser.add_argument(
            '--str_kwargs',
            type=str,
            help='the keyword whose value type is a string.')
        self.parser.add_argument(
            '--num_kwargs',
            type=str,
            help='the keyword whose value type is a number.')
        self.parser.add_argument(
            '--bool_kwargs',
            type=str,
            help='the keyword whose value type is a boolean.')
        self.parser.add_argument(
            '--str_list_kwargs',
            type=str,
            help=
            'the keyword whose value type is a list of strings. please note that this only applies to modifying the classes parameter.'
        )
        self.parser.add_argument(
            '--num_list_kwargs',
            type=str,
            help=
            'the keyword whose value type is a list of numbers. please note that this only applies to modifying the gpus parameter.'
        )

    def parse_kwargs(self, kwargs, kwargs_type, check):
        kwargs_dict = {}
        if kwargs_type in ['str_list', 'num_list']:
            key, value = kwargs.split(sep='=', maxsplit=1)
            if kwargs_type == 'str_list':
                value = value.split(',')
            elif kwargs_type == 'num_list':
                value = [eval(v) for v in value.split(',')]
            if check:
                # check key if exist in the config
                assert key in list(
                    self.config.keys()
                ), 'please check if the keyword argument exists in the configuration.\nkwargs: {}\nvalid: {}'.format(
                    key, list(self.config.keys()))
            exec('kwargs_dict["{}"]={}'.format(key, value))
        else:
            for v in kwargs.split(','):
                key, value = v.split(sep='=', maxsplit=1)
                if check:
                    # check key if exist in the config
                    assert key in list(
                        self.config.keys()
                    ), 'please check if the keyword argument exists in the configuration.\nkwargs: {}\nvalid: {}'.format(
                        key, list(self.config.keys()))
                if kwargs_type == 'str':
                    if value == 'None':
                        exec('kwargs_dict["{}"]={}'.format(key, value))
                    else:
                        exec('kwargs_dict["{}"]="{}"'.format(key, value))
                elif kwargs_type == 'num':
                    exec('kwargs_dict["{}"]={}'.format(key, value))
                elif kwargs_type == 'bool':
                    exec('kwargs_dict["{}"]=bool({})'.format(key, value))
        return kwargs_dict

    def get_kwargs(self, args):
        kwargs_dict = {}
        for key, value in vars(args).items():
            if value is not None and 'kwargs' in key:
                kwargs_type = key.rsplit('_', 1)[0]
                new_dict = self.parse_kwargs(kwargs=value,
                                             kwargs_type=kwargs_type,
                                             check=args.dont_check)
                kwargs_dict.update(new_dict)
        return kwargs_dict

    def set_abspath(self):
        for k, v in self.config.items():
            if type(v) == str and (isfile(v) or isdir(v)):
                self.config[k] = abspath(v)

    def parse(self):
        args = self.parser.parse_args() #將這些self.parser 導入至一個變數裡面。
        if isfile(args.config):
            self.config = load_yaml(filepath=args.config) #導入yaml檔案
        else:
            self.config = {}
        
        self.config['config'] = args.config #新增一個config : MIMII_p6_dB_pump_id_00_normal_abnormal_test_accuracy_0.9565217391304348/config_MIMII_p6_dB_pump_id_00_normal_abnormal.yml
        kwargs_dict = self.get_kwargs(args=args)
        # print(kwargs_dict) #{'mode': 'train'}
        self.config.update(kwargs_dict)
        self.set_abspath()
        return argparse.Namespace(**self.config)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # display each parameter
    for name, value in vars(project_parameters).items():
        print('{:<20}= {}'.format(name, value))
