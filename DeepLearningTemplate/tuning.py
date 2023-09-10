# import
from ray import tune


# class
class BaseTuning:
    def __init__(self) -> None:
        pass

    def parse_hyperparameter_space(self, hyperparameter_space_config):
        hyperparameter_space = {}
        for key, value in hyperparameter_space_config.items():
            for typ, arguments in value.items():
                hyperparameter_space_arguments = []
                for a, b in arguments.items():
                    if type(b) is str:
                        arg = '{}="{}"'.format(a, b)
                    else:
                        arg = '{}={}'.format(a, b)
                    hyperparameter_space_arguments.append(arg)
                hyperparameter_space_arguments = ','.join(
                    hyperparameter_space_arguments)
                arguments = hyperparameter_space_arguments
                hyperparameter_space[key] = eval('tune.{}({})'.format(
                    typ, arguments))
        return hyperparameter_space

    def get_tuning_parameters(self, hyperparameter_space, project_parameters):
        for k, v in hyperparameter_space.items():
            if type(v) == str:
                exec('project_parameters.{}="{}"'.format(k, v))
            else:
                exec('project_parameters.{}={}'.format(k, v))
        project_parameters.num_workers = project_parameters.cpu_resources_per_trial
        return project_parameters
