# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.tuning import BaseTuning
from ray import tune
from copy import deepcopy
from typing import Any
import torch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import ray
from functools import partial
from os.path import join
from src.train import Train


# class
class Tuning(BaseTuning):
    def __init__(self, project_parameters, train_class) -> None:
        super().__init__()
        self.train_class = train_class
        self.project_parameters = project_parameters

    def parse_train_result(self, result):
        train_loss = result['train']['test_loss']
        val_loss = result['val']['test_loss']
        return train_loss, val_loss

    def experiment(self, hyperparameter_space, project_parameters):
        if project_parameters.tuning_test:
            sum_of_train_and_val_loss = sum(
                [v for v in hyperparameter_space.values()])
            tune.report(sum_of_train_and_val_loss=sum_of_train_and_val_loss)
        else:
            tuning_parameters = self.get_tuning_parameters(
                hyperparameter_space=hyperparameter_space,
                project_parameters=deepcopy(project_parameters))
            result = self.train_class(
                project_parameters=tuning_parameters).train()
            train_loss, val_loss = self.parse_train_result(result)
            tune.report(train_loss=train_loss,
                        val_loss=val_loss,
                        sum_of_train_and_val_loss=train_loss + val_loss)
        space = hyperparameter_space.copy()
        #the num_workers will set same as cpu_resources_per_trial
        space['num_workers'] = project_parameters.cpu_resources_per_trial
        print(f'hyperparameter_space: {space}')

    def tuning(self) -> Any:
        # NOTE: there is an error that cannot reproduce the best trial if the device is CPU.
        hyperparameter_space = self.parse_hyperparameter_space(
            hyperparameter_space_config=self.project_parameters.
            hyperparameter_space_config)
        gpu_resources_per_trial = self.project_parameters.gpu_resources_per_trial if torch.cuda.is_available(
        ) else 0
        scheduler = ASHAScheduler(metric='sum_of_train_and_val_loss',
                                  mode='min')
        progress_reporter = CLIReporter(metric_columns=[
            'train_loss', 'val_loss', 'sum_of_train_and_val_loss'
        ])
        ray.init(dashboard_host='0.0.0.0')
        tuning_result = tune.run(
            run_or_experiment=partial(
                self.experiment, project_parameters=self.project_parameters),
            config=hyperparameter_space,
            resources_per_trial={
                'cpu': self.project_parameters.cpu_resources_per_trial,
                'gpu': gpu_resources_per_trial
            },
            num_samples=self.project_parameters.num_samples,
            local_dir=join(self.project_parameters.default_root_dir,
                           'ray_results'),
            scheduler=scheduler,
            progress_reporter=progress_reporter)
        best_trial = tuning_result.get_best_trial(
            metric='sum_of_train_and_val_loss', mode='min')
        if not self.project_parameters.tuning_test:
            project_parameters = self.get_tuning_parameters(
                hyperparameter_space=best_trial.config,
                project_parameters=deepcopy(self.project_parameters))
            result = self.train_class(
                project_parameters=project_parameters).train()
            result['tuning'] = tuning_result
        else:
            result = {'tuning': tuning_result}
        print('-' * 80)
        print('best trial name: {}'.format(best_trial))
        print('best trial result: {}'.format(
            best_trial.last_result['sum_of_train_and_val_loss']))
        best_config = [
            '{}: {}'.format(a, b) for a, b in best_trial.config.items()
        ]
        best_config = '\n'.join(best_config)
        print('best trial config:\n\n{}'.format(best_config))
        print('num_workers: {}'.format(
            self.project_parameters.cpu_resources_per_trial))
        ray.shutdown()
        return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # tuning the model
    result = Tuning(project_parameters=project_parameters,
                    train_class=Train).tuning()
