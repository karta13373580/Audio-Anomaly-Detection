#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.model import BaseModel, load_from_checkpoint
import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# from .feature import Extractor
from os.path import isfile
from torchsummary import summary
from einops import rearrange
from timm.models.layers import trunc_normal_
from halonet_pytorch import HaloAttention
from torch import Tensor

#def
def create_model(project_parameters):
    model = UnsupervisedModel(
        optimizers_config=project_parameters.optimizers_config,
        lr=project_parameters.lr,
        lr_schedulers_config=project_parameters.lr_schedulers_config,
        in_chans=project_parameters.in_chans,
        input_height=project_parameters.input_height,
        latent_dim=project_parameters.latent_dim,
        classes=project_parameters.classes,
        generator_feature_dim=project_parameters.generator_feature_dim,
        discriminator_feature_dim=project_parameters.discriminator_feature_dim,
        adversarial_weight=project_parameters.adversarial_weight,
        reconstruction_weight=project_parameters.reconstruction_weight,
        encoding_weight=project_parameters.encoding_weight,
        backbone=project_parameters.backbone,
        cnn_layers=project_parameters.cnn_layers,
        upsample=project_parameters.upsample,
        is_agg=project_parameters.is_agg,
        kernel_size=project_parameters.kernel_size,
        stride=project_parameters.stride,
        dilation=project_parameters.dilation,
        featmap_size=project_parameters.featmap_size,
        device=project_parameters.device)

    if project_parameters.checkpoint_path is not None:
        if isfile(project_parameters.checkpoint_path):
            model = load_from_checkpoint(
                device=project_parameters.device,
                checkpoint_path=project_parameters.checkpoint_path,
                model=model)
        else:
            assert False, 'please check the checkpoint_path argument.\nthe checkpoint_path value is {}.'.format(
                project_parameters.checkpoint_path)
    return model


#class
class Encoder(nn.Module):
    def __init__(self, input_height, in_chans, out_chans, latent_dim,
                 add_final_conv) -> None:
        super().__init__()
        assert input_height % 16 == 0, 'input_height has to be a multiple of 16'
        layers = []
        layers.append(
            nn.Conv2d(in_channels=in_chans,
                      out_channels=out_chans,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        input_height, out_chans = input_height / 2, out_chans
        while input_height > 4:
            in_channels = out_chans
            out_channels = out_chans * 2
            layers.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            out_chans *= 2
            input_height /= 2
        if add_final_conv:
            layers.append(
                nn.Conv2d(in_channels=out_chans,
                          out_channels=latent_dim,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, input_height, in_chans, out_chans, latent_dim) -> None:
        super().__init__()
        assert input_height % 16 == 0, 'input_height has to be a multiple of 16'
        out_chans, target_size = out_chans // 2, 4
        while target_size != input_height:
            out_chans *= 2
            target_size *= 2
        layers = []
        layers.append(
            nn.ConvTranspose2d(in_channels=latent_dim,
                               out_channels=out_chans,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False))
        layers.append(nn.BatchNorm2d(num_features=out_chans))
        layers.append(nn.ReLU(inplace=True))
        target_size = 4
        while target_size < input_height // 2:
            layers.append(
                nn.ConvTranspose2d(in_channels=out_chans,
                                   out_channels=out_chans // 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False))
            layers.append(nn.BatchNorm2d(num_features=out_chans // 2))
            layers.append(nn.ReLU(inplace=True))
            out_chans //= 2
            target_size *= 2
        layers.append(
            nn.ConvTranspose2d(in_channels=out_chans,
                               out_channels=in_chans,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def generate_relative_distance(number_size):
    """return relative distance, (number_size**2, number_size**2, 2)
    """
    indices = torch.tensor(np.array([[x, y] for x in range(number_size) for y in range(number_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    distances = distances + number_size - 1   # shift the zeros postion
    return distances

class Generator(nn.Module):
    def __init__(self, input_height, in_chans, generator_feature_dim,
                 latent_dim) -> None:
        super().__init__()
        self.encoder1 = Encoder(input_height=input_height,
                                in_chans=in_chans,
                                out_chans=generator_feature_dim,
                                latent_dim=latent_dim,
                                add_final_conv=True)
        self.decoder = Decoder(input_height=input_height,
                               in_chans=in_chans,
                               out_chans=generator_feature_dim,
                               latent_dim=latent_dim)
        self.encoder2 = Encoder(input_height=input_height,
                                in_chans=in_chans,
                                out_chans=generator_feature_dim,
                                latent_dim=latent_dim,
                                add_final_conv=True)

        self.channel_attention1 = ChannelAttentionModule(64)
        self.channel_attention2 = ChannelAttentionModule(128)
        self.channel_attention3 = ChannelAttentionModule(256)
        self.channel_attention4 = ChannelAttentionModule(512)
        self.spatial_attention = SpatialAttentionModule()


        # Halo Attention function is :
        # q shape is (fmap//block_size, fmap//block_size, block_size, block_size, c)
        # k,v shape is (fmap//block_size, fmap//block_size, block_size+2*halo_size, block_size+2*halo_size, c)

        # fmap 32*32, block number have (4,4), so each k,v block window size is (16,16) => shape is (4,4,16,16,c)
        # fmap 32*32, block number have (4,4), so each q block window size is (8,8) => shape is (4,4,8,8,c)
        self.attn1 = HaloAttention(
                dim = 64,         # dimension of feature map
                block_size = 8,    # neighborhood block size (feature map must be divisible by this)
                halo_size = 4,     # halo size (block receptive field)
                dim_head = 64,     # dimension of each head
                heads = 4          # number of attention heads
            )
        # fmap 16*16, block number have (4,4), so each k,v block window size is (8,8) => shape is (4,4,8,8,c)
        # fmap 16*16, block number have (4,4), so each q block window size is (4,4) => shape is (4,4,4,4,c)
        self.attn2 = HaloAttention(
                dim = 128,         # dimension of feature map
                block_size = 4,    # neighborhood block size (feature map must be divisible by this)
                halo_size = 2,     # halo size (block receptive field)
                dim_head = 64,     # dimension of each head
                heads = 4          # number of attention heads
            )
        # fmap 8*8, block number have (4,4), so each k,v block window size is (4,4) => shape is (4,4,4,4,c)
        # fmap 8*8, block number have (4,4), so each q block window size is (2,2) => shape is (4,4,2,2,c)
        self.attn3 = HaloAttention(
                dim = 256,         # dimension of feature map
                block_size = 2,    # neighborhood block size (feature map must be divisible by this)
                halo_size = 1,     # halo size (block receptive field)
                dim_head = 64,     # dimension of each head
                heads = 4          # number of attention heads
            )

    def forward(self, x):

        #encoder
        #input 4 64*64
        conv1 = self.encoder1.layers[0](x)
        leakyrelu1 = self.encoder1.layers[1](conv1) #first channel 64, 32*32
        leakyrelu1 = self.channel_attention1(leakyrelu1) * leakyrelu1
        # leakyrelu1 = self.spatial_attention(leakyrelu1) * leakyrelu1
        leakyrelu1 = self.attn1(leakyrelu1)
        
        conv2 = self.encoder1.layers[2](leakyrelu1)
        e_bn2 = self.encoder1.layers[3](conv2)
        leakyrelu2 = self.encoder1.layers[4](e_bn2) #second channel 128, 16*16
        leakyrelu2 = self.channel_attention2(leakyrelu2) * leakyrelu2
        # leakyrelu2 = self.spatial_attention(leakyrelu2) * leakyrelu2
        leakyrelu2 = self.attn2(leakyrelu2)
        
        conv3 = self.encoder1.layers[5](leakyrelu2)
        e_bn3 = self.encoder1.layers[6](conv3)
        leakyrelu3 = self.encoder1.layers[7](e_bn3) #third channel 256, 8*8
        leakyrelu3 = self.channel_attention3(leakyrelu3) * leakyrelu3
        # leakyrelu3 = self.spatial_attention(leakyrelu3) * leakyrelu3
        leakyrelu3 = self.attn3(leakyrelu3)
    
        conv4 = self.encoder1.layers[8](leakyrelu3)
        e_bn4 = self.encoder1.layers[9](conv4)
        leakyrelu4 = self.encoder1.layers[10](e_bn4) #fourth channel 512, 4*4
        # leakyrelu4 = self.channel_attention4(leakyrelu4) * leakyrelu4
        # leakyrelu4 = self.spatial_attention(leakyrelu4) * leakyrelu4

        latent1 = self.encoder1.layers[11](leakyrelu4) # xhat channel 256, 1*1

        #decoder
        #input 256 1*1
        convt1 = self.decoder.layers[0](latent1)
        d_bn1 = self.decoder.layers[1](convt1)
        relu1 = self.decoder.layers[2](d_bn1) # channel 512, 4*4

        res1 = relu1 + leakyrelu4

        convt2 = self.decoder.layers[3](res1)
        d_bn2 = self.decoder.layers[4](convt2)
        relu2 = self.decoder.layers[5](d_bn2) # channel 256, 8*8

        res2 = relu2 + leakyrelu3

        convt3 = self.decoder.layers[6](res2)
        d_bn3 = self.decoder.layers[7](convt3)
        relu3 = self.decoder.layers[8](d_bn3) # channel 128, 16*16

        res3 = relu3 + leakyrelu2

        convt4 = self.decoder.layers[9](res3)
        d_bn4 = self.decoder.layers[10](convt4)
        relu4 = self.decoder.layers[11](d_bn4) # channel 64, 32*32

        res4 = relu4 + leakyrelu1

        x_hat = self.decoder.layers[12](res4) # x channel 4, 64*64
        
        latent2 = self.encoder2(x_hat)
        return x_hat, latent1, latent2

class Discriminator(nn.Module):
    def __init__(self, input_height, in_chans, discriminator_feature_dim,
                 latent_dim) -> None:
        super().__init__()
        layers = Encoder(input_height=input_height,
                         in_chans=in_chans,
                         out_chans=discriminator_feature_dim,
                         latent_dim=latent_dim,
                         add_final_conv=True)

        layers = list(layers.layers.children())
        self.extractor = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.activation_function = nn.Sigmoid()

    def forward(self, x):
        features = self.extractor(x)
        y = self.activation_function(self.classifier(features))
        # y = self.classifier(features)
        return y, features

#==================================================CBAM=======================================================

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        #使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        #map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

#==================================================CBAM=======================================================

#==================================polyloss=========================================

# class Poly1CrossEntropyLoss(nn.Module):
#     def __init__(self,
#                  num_classes: int,
#                  epsilon: float = 1.0,
#                  reduction: str = "mean",
#                  weight: Tensor = None):
#         """
#         Create instance of Poly1CrossEntropyLoss
#         :param num_classes:
#         :param epsilon:
#         :param reduction: one of none|sum|mean, apply reduction to final loss tensor
#         :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
#         """
#         super(Poly1CrossEntropyLoss, self).__init__()
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.reduction = reduction
#         self.weight = weight
#         return

#     def forward(self, logits, labels):
#         """
#         Forward pass
#         :param logits: tensor of shape [N, num_classes]
#         :param labels: tensor of shape [N]
#         :return: poly cross-entropy loss
#         """


#         labels = labels.mean(1)

#         logits = torch.tensor(logits, dtype=torch.float32)
#         labels = torch.tensor(labels, dtype=torch.long)

#         labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device, dtype=logits.dtype)
#         pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
#         CE = F.cross_entropy(input=logits,
#                              target=labels,
#                              reduction='none',
#                              weight=self.weight)
#         poly1 = CE + self.epsilon * (1 - pt)
#         if self.reduction == "mean":
#             poly1 = poly1.mean()
#         elif self.reduction == "sum":
#             poly1 = poly1.sum()
#         poly1 = torch.tensor(poly1, requires_grad=True)
#         return poly1

#===================================================================================

class UnsupervisedModel(BaseModel):
    def __init__(self, optimizers_config, lr, lr_schedulers_config, in_chans,
                 input_height, latent_dim, classes, generator_feature_dim,
                 discriminator_feature_dim, adversarial_weight,
                 reconstruction_weight, encoding_weight, backbone, cnn_layers,
                 upsample, is_agg, kernel_size, stride, dilation, featmap_size, device) -> None:
        super().__init__(optimizers_config=optimizers_config,
                         lr=lr,
                         lr_schedulers_config=lr_schedulers_config)

        self.generator = Generator(
            input_height=input_height,
            in_chans=in_chans,
            generator_feature_dim=generator_feature_dim,
            latent_dim=latent_dim)

        self.discriminator = Discriminator(
            input_height=input_height,
            in_chans=in_chans,
            discriminator_feature_dim=discriminator_feature_dim,
            latent_dim=latent_dim)

        # self.extractor = Extractor(
        #     backbone=backbone,
        #     cnn_layers=cnn_layers,
        #     upsample=upsample,
        #     is_agg=is_agg,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     dilation=dilation,
        #     featmap_size=featmap_size,
        #     device=device)

        self.adversarial_weight = adversarial_weight
        self.reconstruction_weight = reconstruction_weight
        self.encoding_weight = encoding_weight
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.classes = classes
        self.stage_index = 0
        # self.polyloss = Poly1CrossEntropyLoss(num_classes=256)

    def configure_optimizers(self):
        optimizers_g = self.parse_optimizers(
            params=self.generator.parameters())
        optimizers_d = self.parse_optimizers(
            params=self.discriminator.parameters())
                
        if self.lr_schedulers_config is not None:
            lr_schedulers_g = self.parse_lr_schedulers(optimizers=optimizers_g)
            lr_schedulers_d = self.parse_lr_schedulers(optimizers=optimizers_d)
            return [optimizers_g[0],
                    optimizers_d[0]], [lr_schedulers_g[0], lr_schedulers_d[0]]
        else:
            return [optimizers_g[0], optimizers_d[0]]

    def weights_init(self, module):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def forward(self, x):
        # x = self.extractor(input)
        x_hat, latent1, latent2 = self.generator(x)
        loss = F.l1_loss(input=latent1, target=latent2, reduction='none')
        loss = loss.mean(dim=(1, 2, 3))
        return loss, x_hat

    def shared_step(self, batch):
        x, _ = batch
        # x = self.extractor(input)
        # print(x.shape)
        prob_x, feat_x = self.discriminator(x)
        x_hat, latent1, latent2 = self.generator(x)
        return x, x_hat, latent1, latent2, prob_x, feat_x

    def training_step(self, batch, batch_idx, optimizer_idx):
        torch.use_deterministic_algorithms(False)
        x, x_hat, latent1, latent2, prob_x, feat_x = self.shared_step(
            batch=batch)
        if optimizer_idx == 0:  # generator
            prob_x_hat, feat_x_hat = self.discriminator(x_hat)
            adv_loss = self.l2_loss(feat_x_hat,
                                    feat_x) * self.adversarial_weight
            con_loss = self.l1_loss(x_hat, x) * self.reconstruction_weight
            enc_loss = self.l2_loss(latent2, latent1) * self.encoding_weight
            g_loss = enc_loss + con_loss + adv_loss
            self.log('train_loss',
                     g_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            self.log('train_loss_generator',
                     g_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            return g_loss
            
        if optimizer_idx == 1:  #discriminator
            prob_x_hat, feat_x_hat = self.discriminator(x_hat.detach())
            real_loss = self.bce_loss(prob_x, torch.ones_like(input=prob_x))
            fake_loss = self.bce_loss(prob_x_hat,
                                      torch.zeros_like(input=prob_x_hat))
            d_loss = (real_loss + fake_loss) * 0.5

#==================================polyloss=========================================

            # real_loss = self.polyloss(prob_x, torch.ones_like(input=prob_x))
            # fake_loss = self.polyloss(prob_x_hat,
            #                           torch.zeros_like(input=prob_x_hat))
            # d_loss = (real_loss + fake_loss) * 0.5

#===================================================================================

            # real_loss = torch.mean((prob_x-torch.ones_like(input=prob_x))**2)
            # fake_loss = torch.mean((prob_x_hat-torch.zeros_like(input=prob_x_hat))**2)
            # d_loss = (real_loss + fake_loss) * 0.5

            self.log('train_loss_discriminator',
                     d_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        x, x_hat, latent1, latent2, prob_x, feat_x = self.shared_step(
            batch=batch)
        # generator
        prob_x_hat, feat_x_hat = self.discriminator(x_hat)
        adv_loss = self.l2_loss(feat_x_hat, feat_x) * self.adversarial_weight
        con_loss = self.l1_loss(x_hat, x) * self.reconstruction_weight
        enc_loss = self.l2_loss(latent2, latent1) * self.encoding_weight
        g_loss = enc_loss + con_loss + adv_loss
        # discriminator
        prob_x_hat, feat_x_hat = self.discriminator(x_hat.detach())
        real_loss = self.bce_loss(prob_x, torch.ones_like(input=prob_x))
        fake_loss = self.bce_loss(prob_x_hat,
                                  torch.zeros_like(input=prob_x_hat))
        d_loss = (real_loss + fake_loss) * 0.5

#==================================polyloss=========================================
        
        # real_loss = self.polyloss(prob_x, torch.ones_like(input=prob_x))
        # fake_loss = self.polyloss(prob_x_hat,
        #                             torch.zeros_like(input=prob_x_hat))
        # d_loss = (real_loss + fake_loss) * 0.5

#===================================================================================        

        # real_loss = torch.mean((prob_x-torch.ones_like(input=prob_x))**2)
        # fake_loss = torch.mean((prob_x_hat-torch.zeros_like(input=prob_x_hat))**2)
        # d_loss = (real_loss + fake_loss) * 0.5        

        if d_loss.item() < 1e-5:
            self.discriminator.apply(self.weights_init)
        self.log('val_loss', g_loss)
        self.log('val_loss_generator', g_loss, prog_bar=True)
        self.log('val_loss_discriminator', d_loss, prog_bar=True)
        self.log('val_loss_extractor', g_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input, y = batch
        loss = self.forward(input)[0]
        self.log('test_loss', loss)
        loss_step = loss.cpu().data.numpy()
        # print(loss_step)
        y_step = y.cpu().data.numpy()
        return {'y': y_step, 'loss': loss_step}

    def calculate_threshold(self, x1, x2):
        #estimate kernel density
        kde1 = gaussian_kde(x1)
        kde2 = gaussian_kde(x2)

        #generate the data
        xmin = min(x1.min(), x2.min())
        xmax = max(x1.max(), x2.max())
        dx = 0.2 * (xmax - xmin)
        xmin -= dx
        xmax += dx
        data = np.linspace(xmin, xmax, len(x1))

        #get density with data
        kde1_x = kde1(data)
        kde2_x = kde2(data)

        #calculate intersect
        idx = np.argwhere(np.diff(np.sign(kde1_x - kde2_x))).flatten()
        return data[idx]

    def calculate_confusion_matrix(self, y, loss):
        normal_score = loss[y == self.classes.index('normal')]
        abnormal_score = loss[y == self.classes.index('abnormal')]

        threshold = self.calculate_threshold(x1=normal_score,
                                             x2=abnormal_score)
        max_auc, best_threshold = 0, 0
        
        for v in threshold:
            y_pred = np.where(loss < v, self.classes.index('normal'),
                              self.classes.index('abnormal'))
            confusion_matrix = pd.DataFrame(metrics.confusion_matrix(
                y_true=y, y_pred=y_pred,
                labels=list(range(len(self.classes)))),
                                            index=self.classes,
                                            columns=self.classes)
            accuracy = np.diagonal(
                confusion_matrix).sum() / confusion_matrix.values.sum()
            auc_score = metrics.roc_auc_score(y, y_pred)   
            if auc_score > max_auc:
                max_auc = auc_score
                best_threshold = v


        y_pred = np.where(loss < best_threshold, self.classes.index('normal'),
                          self.classes.index('abnormal'))
        confusion_matrix = pd.DataFrame(metrics.confusion_matrix(
            y_true=y, y_pred=y_pred, labels=list(range(len(self.classes)))),
                                        index=self.classes,
                                        columns=self.classes)
        accuracy = np.diagonal(
            confusion_matrix).sum() / confusion_matrix.values.sum()
        auc_score = metrics.roc_auc_score(y, y_pred)

        print(f'best_threshold: {best_threshold}')
        print(f'auc_score: {auc_score}')
#=======================================================================

        # cm = metrics.confusion_matrix(y, y_pred)
        # ax = sns.heatmap(cm,annot=True,fmt='g',xticklabels=['normal', 'abnormal'],yticklabels=['normal', 'abnormal'])
        # ax.set_title(f'confusion matrix\nbest_threshold: {best_threshold}')
        # ax.set_xlabel('predict')
        # ax.set_ylabel('true')
        # plt.savefig(f"./result_photo/Halo_mimii.png")
        # plt.show()

#=======================================================================
        return confusion_matrix, accuracy, best_threshold

    def test_epoch_end(self, test_outs):
        stages = ['train', 'val', 'test']
        print('\ntest the {} dataset'.format(stages[self.stage_index]))
        print('the {} dataset confusion matrix:'.format(
            stages[self.stage_index]))
        y = np.concatenate([v['y'] for v in test_outs])
        loss = np.concatenate([v['loss'] for v in test_outs])
        figure = plt.figure(figsize=[11.2, 6.3])
        plt.title(stages[self.stage_index])
        for idx, v in enumerate(self.classes):
            score = loss[y == idx]
            sns.kdeplot(score, label=v)
        plt.xlabel(xlabel='Loss')
        plt.legend()
        plt.close()
        self.logger.experiment.add_figure(
            '{} loss density'.format(stages[self.stage_index]), figure,
            self.current_epoch)
        if stages[self.stage_index] == 'test':
            confusion_matrix, accuracy, best_threshold = self.calculate_confusion_matrix(
                y=y, loss=loss)
            print(confusion_matrix)
            plt.figure(figsize=[11.2, 6.3])
            plt.title('{}\nthreshold: {}\naccuracy: {}'.format(
                stages[self.stage_index], best_threshold, accuracy))
            figure = sns.heatmap(data=confusion_matrix,
                                 cmap='Spectral',
                                 annot=True,
                                 fmt='g').get_figure()
            plt.yticks(rotation=0)
            plt.ylabel(ylabel='Actual class')
            plt.xlabel(xlabel='Predicted class')
            plt.close()
            self.logger.experiment.add_figure(
                '{} confusion matrix'.format(stages[self.stage_index]), figure,
                self.current_epoch)
            self.log('test_accuracy', accuracy)
        self.stage_index += 1


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    summary(model=model,
            input_size=(project_parameters.in_chans,
                        project_parameters.input_height,
                        project_parameters.input_height),
            device='cpu')

    # create input data
    x = torch.ones(project_parameters.batch_size, project_parameters.in_chans,
                   project_parameters.input_height,
                   project_parameters.input_height)

    # get model output
    loss, x_hat = model(x)

    # display the dimension of input and output
    print('the dimension of input: {}'.format(x.shape))
    print('the dimension of output: {}'.format(x_hat.shape))
