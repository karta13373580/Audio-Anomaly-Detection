# Audio-Anomaly-Detection
<https://hdl.handle.net/11296/xdh5fm>
### 使用通道與空間注意力於GANomaly進行非監督異常工業設備聲音檢測
##### Unsupervised Abnormal Industrial Equipment Sounds Detection Using Channel and Spatial Attention on GANomaly

## 方法
<img src="https://github.com/karta13373580/Audio-Anomaly-Detection/blob/main/result_photo/github_photo/1.PNG">

在工業製造過程中，機器運轉的時候會產生聲音，其中所出現的異常聲響可能表明內部零件出現損壞，進而影響生產效率。因此機器異常聲音的識別是相當重要的事情。本研究中，我們提出基於GANomaly網路方法，該模型結合GAN與Autoencoder的思想，使訓練過程更加穩定。並透過比較正常與異常聲音之間的差異，進而檢測出設備是否出現問題。  
前處理上將聲音轉換成同時帶有時間與頻率特徵的梅爾頻譜，並在GANomaly中的Encoder添加Channel Attention Module與Halo Attention，增加通道之間的關聯性，以及強化梅爾頻譜的空間特徵，藉此增加經過編碼後所得到的潛在向量資訊，接著引入Skip Connection 架構，捕捉Encoder下採樣過程中的多尺度影像特徵，將這些資訊輸入至Decoder，使其能夠重建更高品質的影像。最後模型計算出來的正常結果透過Kernel Density Estimation映射於高斯分布，找出較佳的閥值。實驗結果在MIMII工業聲音的公開資料集進行檢測，在風扇、泵、滑軌和閥門類別平均AUC分別獲得了84.94%、86.79%、75.24%、73.62%。

## 實驗結果
MIMII資料集: <https://hdl.handle.net/11296/xdh5fm>

| Machine ID | Fan | Pump | Slider | Valve | Average |
| :----: | :----: | :----: | :----: | :----: | :----: |
| 00 | 73.18% | 97.05% | 96.82% | 66.43% | 84.94% |
| 02 | 90.03% | 76.47% | 82.01% | 98.33% | 86.79% |
| 04 | 80.31% | 86.85% | 59.60% | 55.83% | 75.24% |
| 06 | 96.25% | 86.79% | 62.55% | 73.92% | 73.62% |

## 與其他論文實驗比較
| Model | Fan | Pump | Slider | Valve |
| :----: | :----: | :----: | :----: | :----: |
| Autoencoder | 65.83% | 72.89% | 84.76% | 66.28% |
| LSTM- Autoencoder | 67.32% | 73.94% | 84.99% | 67.82% |
| Dictionary Learning-Autoencoder | 79.60% | 84.91% | 82.00% | 72.33% |
| Contrastive Learning | 80.11% | 70.12% | 77.43% | 84.17% |
| Baseline-GANomaly | 80.34% | 83.90% | 72.70% | 68.51% |
| Proposed | 84.94% | 86.79% | 75.24% | 73.62% |

## 與其他論文實驗比較
*1
*2
```
pip install -r requirements.txt
```
