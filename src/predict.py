# import
from src.project_parameters import ProjectParameters
from src.model import create_model
import torch
from DeepLearningTemplate.data_preparation import parse_transforms, AudioLoader
from DeepLearningTemplate.predict import AudioPredictDataset
from typing import TypeVar, Any
import pathlib
T_co = TypeVar('T_co', covariant=True)
from os.path import isfile
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from ptflops import get_model_complexity_info
import os


# class
class AudioPredictDataset(AudioPredictDataset):
    def __init__(self, root, loader, transform) -> None:
        super().__init__(root, loader, transform)

    def __getitem__(self, index) -> T_co:
        path = self.samples[index]
        sample = self.get_sample(path=path)
        sample = sample.mean(0)[None]
        if self.transform is not None:
            sample = self.transform(sample)
            # sample = torchaudio.functional.amplitude_to_DB(sample, 10.0, 1e-10, torch.log10(max(sample.max(), 1e-10)), None)
        c, f, t = sample.shape
        sample = torch.cat(
            [sample[:, :, idx:idx + f] for idx in range(0, t, f)])
        return sample


class Predict:
    def __init__(self, project_parameters) -> None:
        self.model = create_model(project_parameters=project_parameters).eval()
        self.flops, self.params = get_model_complexity_info(self.model, (project_parameters.in_chans,project_parameters.input_height,project_parameters.input_height), as_strings = True, print_per_layer_stat = False)
        if project_parameters.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.device = project_parameters.device
        self.batch_size = project_parameters.batch_size
        self.num_workers = project_parameters.num_workers
        self.classes = project_parameters.classes
        self.loader = AudioLoader(sample_rate=project_parameters.sample_rate)
        self.in_chans = project_parameters.in_chans
        self.threshold = project_parameters.threshold

    def predict(self, inputs) -> Any:
        result = []
        # fake_samples = []

        if isfile(path=inputs):
            # predict the file
            sample = self.loader(path=inputs)
            in_chans, _ = sample.shape
            if in_chans != 1:
                sample = sample.mean(0)
                sample = torch.cat(
                    [sample[None] for idx in range(1)])
            # the transformed sample dimension is (1, in_chans, freq, time)
            sample = self.transform(sample)
            c, f, t = sample.shape
            sample = torch.cat(
                [sample[:, :, idx:idx + f] for idx in range(0, t, f)])
            sample = sample[None]
            if self.device == 'cuda' and torch.cuda.is_available():
                sample = sample.cuda()
            with torch.no_grad():
                score, sample_hat = self.model(sample)
                #sample_hat dimension is (batch_size, in_chans, freq, time)
                # sample_hat = torch.cat([
                #     sample_hat[:, idx:idx + c, ...]
                #     for idx in range(0, sample_hat.shape[1], c)
                # ], -1)
                result.append([score.item()])
                # fake_samples.append(sample_hat.cpu().data.numpy())
        else:
            # predict the file from folder
            dataset = AudioPredictDataset(root=inputs,
                                        loader=self.loader,
                                        transform=self.transform)
            pin_memory = True if self.device == 'cuda' and torch.cuda.is_available(
            ) else False
            data_loader = DataLoader(dataset=dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    pin_memory=pin_memory)
            with torch.no_grad(): #torch.no_grad()，不要梯度傳播，可以切換成測試模式，去執行model.py 裡的測試模型，也就是 test_step 裡的 self.forward，只會跑完模型部分，其他要自己寫
                for sample in tqdm(data_loader):
                    if self.device == 'cuda' and torch.cuda.is_available():
                        sample = sample.cuda()
                    score, sample_hat = self.model(sample)
                    # sample_hat dimension is (batch_size, in_chans, freq, time)
                    # c = 1
                    # sample_hat = torch.cat([
                    #     sample_hat[:, idx:idx + c, ...]
                    #     for idx in range(0, sample_hat.shape[1], c)
                    # ], -1)

                    result.append(score.tolist())
                    # fake_samples.append(sample_hat.cpu().data.numpy())
            result = np.concatenate(result, 0)

            filenames = []
            for path in pathlib.Path(inputs).iterdir():
                basename = os.path.basename(path)
                filename = os.path.splitext(basename)[0]
                filenames.append(filename)
            for i in range(len(result)):
                #normal
                if result[i] < self.threshold: 
                    print(f"{filenames[i]} is normal")
                #abnormal
                else:  
                    print(f"warning: {filenames[i]} is abnormal")

        return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict file
    result = Predict(project_parameters=project_parameters).predict(
        inputs=project_parameters.root)
