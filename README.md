# Audio-Anomaly-Detection
<https://hdl.handle.net/11296/xdh5fm>
### 使用通道與空間注意力於GANomaly進行非監督異常工業設備聲音檢測
##### Unsupervised Abnormal Industrial Equipment Sounds Detection Using Channel and Spatial Attention on GANomaly

## 方法
<div align=center>
<img src="https://github.com/karta13373580/Audio-Anomaly-Detection/blob/main/result_photo/github_photo/1.PNG">
</div>

在工業製造過程中，機器運轉的時候會產生聲音，其中所出現的異常聲響可能表明內部零件出現損壞，進而影響生產效率。因此機器異常聲音的識別是相當重要的事情。本研究中，我們提出基於GANomaly網路方法，該模型結合GAN與Autoencoder的思想，使訓練過程更加穩定。並透過比較正常與異常聲音之間的差異，進而檢測出設備是否出現問題。
前處理上將聲音轉換成同時帶有時間與頻率特徵的梅爾頻譜，並在GANomaly中的Encoder添加Channel Attention Module與Halo Attention，增加通道之間的關聯性，以及強化梅爾頻譜的空間特徵，藉此增加經過編碼後所得到的潛在向量資訊，接著引入Skip Connection 架構，捕捉Encoder下採樣過程中的多尺度影像特徵，將這些資訊輸入至Decoder，使其能夠重建更高品質的影像。最後模型計算出來的正常結果透過Kernel Density Estimation映射於高斯分布，找出較佳的閥值。實驗結果顯示在我們所蒐集的船舶資料集中，靠岸與出海類別的平均AUC各自取得了95.99%與97.25%，且為了進一步驗證模型的效能，同時在工業聲音的公開資料集進行檢測，在風扇、泵、滑軌和閥門類別平均AUC分別獲得了84.94%、86.79%、75.24%、73.62%。
