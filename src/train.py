# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.train import BaseTrain
from src.data_preparation import create_datamodule
from ptflops import get_model_complexity_info
from src.model import create_model
from typing import Any
from shutil import copy2
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
# class
class Train(BaseTrain):
    def __init__(self, project_parameters) -> None:
        super().__init__(seed=project_parameters.seed)

        self.datamodule = create_datamodule(project_parameters=project_parameters)
        self.model = create_model(project_parameters=project_parameters).cuda()
        
        self.flops, self.params = get_model_complexity_info(self.model, (project_parameters.in_chans,project_parameters.input_height,project_parameters.input_height), as_strings = True, print_per_layer_stat = False)
        print('Flops:  ' + self.flops)
        print('Params: ' + self.params)

        self.trainer = self.create_trainer(
            early_stopping=project_parameters.early_stopping,
            patience=project_parameters.patience,
            device=project_parameters.device,
            default_root_dir=project_parameters.default_root_dir,
            gpus=project_parameters.gpus,
            precision=project_parameters.precision,
            max_epochs=project_parameters.max_epochs)
            
        self.config = project_parameters.config

    def train(self) -> Any:
        self.trainer.fit(model=self.model, datamodule=self.datamodule)
        self.datamodule.setup(stage='test')
        dataloaders_dict = {
            'train': self.datamodule.train_dataloader(),
            'val': self.datamodule.val_dataloader(),
            'test': self.datamodule.test_dataloader()
        }
        result = {'trainer': self.trainer, 'model': self.model}
        for stage, dataloader in dataloaders_dict.items():
            result[stage] = self.trainer.test(dataloaders=dataloader,
                                              ckpt_path='best')[0]
        #copy config to save directory
        copy2(src=self.config, dst=self.trainer.logger.log_dir)
        return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # train the model
    result = Train(project_parameters=project_parameters).train()
