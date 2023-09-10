# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.data_preparation import MySPEECHCOMMANDS, MyAudioFolder, AudioLightningDataModule
from typing import Union, Optional, Tuple, Callable, Any
from pathlib import Path
from torch import Tensor
import os
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as transforms
import librosa
import numpy as np

#def
def create_datamodule(project_parameters):
    if project_parameters.predefined_dataset:
        dataset_class = eval('My{}'.format(project_parameters.predefined_dataset))
    else:
        dataset_class = MyAudioFolder
    return AudioLightningDataModule(
        root=project_parameters.root,
        predefined_dataset=project_parameters.predefined_dataset,
        classes=project_parameters.classes,
        max_samples=project_parameters.max_samples,
        batch_size=project_parameters.batch_size,
        num_workers=project_parameters.num_workers,
        device=project_parameters.device,
        transforms_config=project_parameters.transforms_config,
        target_transforms_config=project_parameters.target_transforms_config,
        sample_rate=project_parameters.sample_rate,
        dataset_class=dataset_class)


#class
class MySPEECHCOMMANDS(MySPEECHCOMMANDS):
    def __init__(self,
                 root: Union[str, Path],
                 loader,
                 transform,
                 target_transform,
                 download: bool = False,
                 subset: Optional[str] = None) -> None:
        super().__init__(root,
                         loader,
                         transform,
                         target_transform,
                         download=download,
                         subset=subset)
        normal_class = 'dog'
        # the index of normal is 0 and abnormal is 1
        self.class_to_idx = {v: 1 for v in self.class_to_idx.keys()}
        self.class_to_idx[normal_class] = 0
        self.classes = ['normal', 'abnormal']
        if subset != 'testing':
            # training and validation dataset only have normal class
            self._walker = [f for f in self._walker if normal_class in f]

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        sample, target = super().__getitem__(n)
        # convert the range of the sample to 0~1
        sample = (sample - sample.min()) / (sample.max() - sample.min())
        return sample, target


class MyAudioFolder(MyAudioFolder):
    def __init__(self,
                 root: str,
                 loader: Callable[[str], Any],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root,
                         loader,
                         transform=transform,
                         target_transform=target_transform)
        self.classes = ['normal', 'abnormal']
        self.class_to_idx = {k: idx for idx, k in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        # mean_sample = torch.mean(sample, dim=1)
        # std_sample = torch.std(sample, dim=1)
        # sample = F.normalize(sample, p=1, dim=1)
        
        sample = sample.mean(0)[None]
        if self.transform is not None:
            sample = self.transform(sample)
            # print(sample.shape)
            # sample = torchaudio.functional.amplitude_to_DB(sample, 10.0, 1e-10, torch.log10(max(sample.max(), 1e-10)), None)
            
            # sample = sample.numpy()[0,:]
            # sample = librosa.feature.melspectrogram(y=sample, sr=16000, n_mels=64, n_fft=1024, hop_length=512)
            # sample = librosa.power_to_db(sample, ref=np.max)
            # sample = np.expand_dims(sample, 0)
            # sample = torch.from_numpy(sample)
            # sample = sample[...,:256]
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        path, _ = self.samples[index]
        relpath = os.path.relpath(path, self.root)
        label, filename = os.path.split(relpath)
        target = self.class_to_idx[label]
        c, f, t = sample.shape
        # print(sample[:1, :, :, :])
        sample = torch.cat(
            [sample[:, :, idx:idx + f] for idx in range(0, t, f)])
        # print(sample[:, :, :])
        # print(sample)
        
        return sample, target


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    # prepare data
    datamodule.prepare_data()

    # set up data
    datamodule.setup()

    # get train, validation, test dataset
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset

    # get the first sample and target in the train dataset
    x, y = train_dataset[0]

    # display the dimension of sample and target
    print('the dimension of sample: {}'.format(x.shape))
    print(
        'the dimension of target: {}'.format(1 if type(y) == int else y.shape))
