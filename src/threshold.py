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
import torchaudio
import torchvision.transforms as transforms
import librosa
from matplotlib.pyplot import MultipleLocator

def plot_melspectrum(origin_samples, fake_samples):
    fake_samples = fake_samples[50,:,:,:]
    origin_samples = origin_samples[50,:,:,:]

    plt.figure()

    plt.subplot(211)
    plt.title("origin data")
    plt.imshow(origin_samples.detach()[0, :, :].numpy(), origin = 'lower')

    plt.subplot(212)
    plt.title("generate data")
    plt.imshow(fake_samples.detach()[0, :, :].numpy(), origin = 'lower')

    # plt.show()


def plot_confusion_matrix(label, y_pred, best_threshold, accuracy, auc_score):
        plt.figure(figsize = (11.5, 10))
        cm = metrics.confusion_matrix(label, y_pred)
        sns.set(font_scale = 1.5)
        ax = sns.heatmap(cm,annot=True,fmt='g',xticklabels=['normal', 'abnormal'],yticklabels=['normal', 'abnormal'])
        ax.set_title(f'confusion matrix\nbest_threshold: {best_threshold}\naccuracy: {accuracy}\nauc_score: {auc_score}')
        ax.set_xlabel('predict')
        ax.set_ylabel('true')
        # plt.savefig(f"./result_photo/Halo_AUC_{auc_score}.png")
        # plt.show()

def calculate_threshold(x1, x2):
    #estimate kernel density
    kde1 = gaussian_kde(x1)
    kde2 = gaussian_kde(x2)

    #generate the data
    xmin = min(x1.min(), x1.min())
    xmax = max(x1.max(), x1.max())
    dx = 0.2 * (xmax - xmin)
    xmin -= dx
    xmax += dx
    data = np.linspace(xmin, xmax, len(x1))

    #get density with data
    kde1_x = kde1(data)
    kde2_x = kde2(data)

    #calculate intersect
    # idx = np.argwhere(np.diff(np.sign(kde1_x - kde2_x))).flatten()

    # print(np.sign(kde1_x - kde2_x))
    # print(np.diff(np.sign(kde1_x - kde2_x)))
    # print("idx", idx)

    # plt.figure(figsize=(20,10))
    # plt.plot(kde1_x, color="skyblue")

    # plt.plot(kde2_x, color="red")

    # plt.plot(data[idx], kde2_x[idx], 'ko')
    plt.fill_between(data, kde1_x, color="skyblue", alpha=0.4)
    # plt.fill_between(data, kde2_x, color="red", alpha=0.4)
    # plt.title("Density Plot of the data")
    # plt.xlabel("new generate data")
    # plt.ylabel("get estimate kernel density with data")
    # plt.show()

    # sns.displot(kde1_x, bins=50, kde=False)
    # x_major_locator = MultipleLocator(0.01)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.show()

    # y is The number of occurrences, x is The value of the actual data
    # plt.hist(x1,202, alpha = 0.4, color="skyblue")
    # plt.hist(x2,143, alpha = 0.4, color="red")
    plt.show()

    return data[idx]

def calculate_TP(y, y_pred): 
    tp = 0 
    for i, j in zip(y, y_pred): 
        if i == j == 1: 
            tp += 1 
    return tp 

def calculate_TN(y, y_pred): 
    tn = 0 
    for i, j in zip(y, y_pred): 
        if i == j == 0: 
            tn += 1 
    return tn 

def calculate_FP(y, y_pred): 
    fp = 0 
    for i, j in zip(y, y_pred): 
        if i == 0 and j == 1: 
            fp += 1 
    return fp 

def calculate_FN(y, y_pred): 
    fn = 0 
    for i, j in zip(y, y_pred): 
        if i == 1 and j == 0: 
            fn += 1 
    return fn

def calculate(label, y_pred):

    tp = calculate_TP(label, y_pred)

    tn = calculate_TN(label, y_pred)

    fp = calculate_FP(label, y_pred)

    fn = calculate_FN(label, y_pred)

    return tp, tn, fp, fn
    
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
            # sample = sample.numpy()[0,:]
            # sample = librosa.feature.melspectrogram(y=sample, sr=16000, n_mels=64, n_fft=1024, hop_length=512)
            # sample = librosa.power_to_db(sample, ref=np.max)
            # sample = np.expand_dims(sample, 0)
            # sample = torch.from_numpy(sample)
            # sample = sample[...,:256]

        c, f, t = sample.shape
        sample = torch.cat(
            [sample[:, :, idx:idx + f] for idx in range(0, t, f)])
        return sample


class Thr:
    def __init__(self, project_parameters) -> None:
        self.model = create_model(project_parameters=project_parameters).eval()
        self.flops, self.params = get_model_complexity_info(self.model, (project_parameters.in_chans,project_parameters.input_height,project_parameters.input_height), as_strings = True, print_per_layer_stat = False)
        if project_parameters.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['test']
        self.device = project_parameters.device
        self.batch_size = project_parameters.batch_size
        self.num_workers = project_parameters.num_workers
        self.classes = project_parameters.classes
        self.loader = AudioLoader(sample_rate=project_parameters.sample_rate)
        self.in_chans = project_parameters.in_chans

    def thr(self, inputs) -> Any:
        result = []
        origin_samples = []
        fake_samples = []

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
                sample_hat = torch.cat([
                    sample_hat[:, idx:idx + c, ...]
                    for idx in range(0, sample_hat.shape[1], c)
                ], -1)
                result.append([score.item()])
                # fake_samples.append(sample_hat.cpu().data.numpy())
        else:
            for idx, species in enumerate(self.classes): #跑兩次模型，一次normal資料夾，一次abnormal資料夾
                # predict the file from folder
                dataset = AudioPredictDataset(root=inputs +'/thr/'+ species,
                                            loader=self.loader,
                                            transform=self.transform)
                pin_memory = True if self.device == 'cuda' and torch.cuda.is_available(
                ) else False
                data_loader = DataLoader(dataset=dataset,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        pin_memory=pin_memory)
                #建立label，正確標籤                      
                data_count = 0
                for path in pathlib.Path(inputs +'/thr/'+ species).iterdir():
                    if path.is_file():
                        data_count += 1
                if species == "normal":
                    normal_label = np.zeros(data_count)
                if species == "abnormal":
                    abnormal_label = np.ones(data_count)

                with torch.no_grad(): #torch.no_grad()，不要梯度傳播，可以切換成測試模式，去執行model.py 裡的測試模型，也就是 test_step 裡的 self.forward，只會跑完模型部分，其他要自己寫
                    for sample in tqdm(data_loader):
                        if self.device == 'cuda' and torch.cuda.is_available():
                            sample = sample.cuda()
                        score, sample_hat = self.model(sample)
                        # sample_hat dimension is (batch_size, in_chans, freq, time)
                        c = 1

                        sample = torch.cat([
                            sample[:, idx:idx + c, ...]
                            for idx in range(0, sample.shape[1], c)
                        ], -1)

                        sample_hat = torch.cat([
                            sample_hat[:, idx:idx + c, ...]
                            for idx in range(0, sample_hat.shape[1], c)
                        ], -1)

                        result.append(score.tolist())
                        origin_samples.append(sample.cpu().data.numpy())
                        fake_samples.append(sample_hat.cpu().data.numpy())

            result = np.concatenate(result, 0)
            origin_samples = np.concatenate(origin_samples, 0)
            origin_samples = torch.from_numpy(origin_samples)           
            fake_samples = np.concatenate(fake_samples, 0)
            fake_samples = torch.from_numpy(fake_samples)
            
            label = np.concatenate((normal_label, abnormal_label))

            #計算大於或小於best_threshold，也就是判斷正常或異常
            normal_score = result[label == 0]
            sns.displot(normal_score, bins=50, kde=False)

            normal_score_mean = np.mean(normal_score)
            normal_score_std = np.std(normal_score)

            print("threshold_mean : ", normal_score_mean)
            print("threshold_std : ", normal_score_std)
            print("threshold_mean+1*std : ", normal_score_mean+1*(normal_score_std))
            print("threshold_mean+2*std : ", normal_score_mean+2*(normal_score_std))
            print("threshold_mean+3*std : ", normal_score_mean+3*(normal_score_std))

            # sns.kdeplot(normal_score)
            plt.show()

            abnormal_score = result[label == 1]
            # sns.displot(abnormal_score, bins=50, kde=False)
            # sns.kdeplot(abnormal_score)
            # plt.show()

            best_threshold = 0.18

            threshold = calculate_threshold(x1=normal_score,
                                        x2=abnormal_score)

            # max_auc, best_threshold = 0, 0
            
            # for v in threshold:
            #     y_pred = np.ones(len(label))
            #     y_pred[ result < v ] = 0

            #     # confusion_matrix = pd.DataFrame(metrics.confusion_matrix(
            #     #     y_true=label, y_pred=y_pred,
            #     #     labels=list(range(len(self.classes)))),
            #     #                                 index=self.classes,
            #     #                                 columns=self.classes)
            #     # accuracy = np.diagonal(
            #     #     confusion_matrix).sum() / confusion_matrix.values.sum()
            #     auc_score = metrics.roc_auc_score(label, y_pred)

            #     if auc_score > max_auc:
            #         max_auc = auc_score
            #         best_threshold = v
            # # accuracy = np.diagonal(confusion_matrix).sum() / confusion_matrix.values.sum()

            y_pred = np.ones(len(label))
            y_pred[ result < best_threshold ] = 0

            print(f'best_threshold: {best_threshold}')   

            #計算tp, tn, fp, fn
            tp, tn, fp, fn = calculate(label, y_pred)
            
            auc_score = metrics.roc_auc_score(label, y_pred)
            accuracy = (tp + tn) / (tp + fp + fn + tn)
            precision =  tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 / ( (1 / precision) + (1 / recall) )
            print(f"auc_score: {auc_score}\naccuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}")  

            print('Flops:  ' + self.flops)
            print('Params: ' + self.params)

            #畫混淆矩陣與準確率
            # plot_melspectrum(origin_samples, fake_samples)
            # plot_confusion_matrix(label, y_pred, best_threshold, accuracy, auc_score)
        # return result, fake_samples
        return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict file
    result = Test(project_parameters=project_parameters).test(
        inputs=project_parameters.root)
