# 2025-NTHU-BCI-final-project

## Group 4  
113062524 林玟萱  
113062608 王彥翔  
110030034 鄭羿羚

## 1. Introduction

BCI (Brain-Computer Interface), this technology holds great potential for assisting individuals with physical disabilities, particularly through the use of MI (Motor Imagery). While MI-BCI systems often require extensive user training, we explore whether DL (Deep Learning) can improve signal recognition, especially for inefficient users, thereby enhancing the overall effectiveness and accessibility of MI-based BCI applications.

## 2. Data Collection

**Dataset:** The BCI Competition IV – Dataset 2a  

| Attribute                  | Description                                |
|---------------------------|--------------------------------------------|
| **Subjects**              | 9 healthy subjects (A01–A09)               |
| **Sessions per Subject**  | 2 (Training + Testing)                     |
| **MI Classes**            | Left hand, Right hand, Both feet, Tongue  |
| **Trials per Session**    | 288 (72 per class)                         |
| **Sampling Rate**         | **250 Hz**                                 |
| **Number of Channels**    | **22 EEG channels**, 3 EOG channels        |
| **Duration per Trial**    | ~7.5 seconds                               |
| **File Format**           | `.gdf` for labels and events     |

**Trial timing:**  
- t = 0 s: Fixation and beep  
- t = 2 s: Arrow cue (1.25 s)  
- t = 2 to 6 s: Motor imagery  
- t = 6 to 7.5 s: Rest
  
 ![Timing Scheme Paradigm](https://raw.githubusercontent.com/orvindemsy/BCICIV2a-FBCSP/d6dce55b4951b8e46bb5e625b060d332101cdd59/img/timing-scheme-paradigm.png)

 

---

##  Source and Citation
- **Website**: [https://www.bbci.de/competition/iv/](https://www.bbci.de/competition/iv/)
- **Owner / Source**: provided by the Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology, (Clemens Brunner, Robert Leeb, Gernot Müller-Putz, Alois Schlögl, Gert Pfurtscheller)







## 3. Model Framework
First, we use EEGLAB for initial data preprocessing.

 <img src="https://github.com/elaine17016/2025-NTHU-BCI-final-project/blob/main/image/Preprocessing%20Pipeline.png?raw=true" width="400px">
 
Next, we apply preprocessing.m in MATLAB to perform advanced preprocessing steps and retrieve the resulting .mat file.

 <img src="https://github.com/elaine17016/2025-NTHU-BCI-final-project/blob/main/image/preprocessing.m%20%E6%B5%81%E7%A8%8B%E5%9C%96.png?raw=true" width="400px">
 
Then, we conduct machine learning. Besides SVM, we also use other models such as KNN, Random Forest, and XGBoost etc.

 <img src="https://github.com/elaine17016/2025-NTHU-BCI-final-project/blob/main/image/ML%20process.png?raw=true" width="400px">

## Results
### 🧠 Why we choose 7-30Hz? Comparison of EEG Frequency Bands on Machine Learning Performance
After using svm.py to train, this table compares classification performance under different filtered frequency bands (Hz), including metrics from training, validation, and testing phases:

| Item / Frequency Band | **1–50 Hz** | **7–30 Hz** | **7–13 Hz** |
|------------------------|-------------|-------------|-------------|
| Original X shape       | (500, 3, 568) | (500, 3, 576) | (500, 3, 576) |
| Transposed X shape     | (568, 3, 500) | (576, 3, 500) | (576, 3, 500) |
| Original y shape       | (568,)        | (576,)        | (576,)        |
| Converted data shape   | (1704, 500)   | (1728, 500)   | (1728, 500)   |
| Train / Test Split     | Train: 1363<br>Test: 341 | Train: 1382<br>Test: 346 | Train: 1382<br>Test: 346 |
| **Avg. Training Acc.** | 0.9765        | **0.9792**    | 0.9410        |
| **Avg. Validation Acc.** | 0.8129      | **0.8589**    | 0.8061        |
| **Final Test Accuracy** | 0.8475      | **0.9133**    | 0.8208        |
| Left Hand Acc. (Validation) | 0.8223 (560/681) | **0.8698 (691/691)** | 0.8307 (574/691) |
| Right Hand Acc. (Validation) | 0.8035 (548/682) | **0.8480 (586/691)** | 0.7815 (540/691) |
| Test Precision          | Left: 0.87<br>Right: 0.83 | Left: 0.91<br>Right: 0.91 | Left: 0.81<br>Right: 0.84 |
| Test Recall             | Left: 0.82<br>Right: 0.87 | Left: 0.91<br>Right: 0.91 | Left: 0.84<br>Right: 0.80 |
| Test F1-Score           | Left: 0.84<br>Right: 0.85 | Left: 0.91<br>Right: 0.91 | Left: 0.82<br>Right: 0.82 |
| Test Support            | 171 / 170     | 173 / 173     | 173 / 173     |

---

##  Recommendation

- The **7–30 Hz** band performs best, with both validation and test accuracy reaching **91%**.
- The **1–50 Hz** range includes high-frequency noise, slightly degrading validation performance.
- The **7–13 Hz** band focuses on alpha rhythms, which may not fully capture the motor imagery signals, resulting in slightly lower accuracy.

> We recommend using the **7–30 Hz** frequency band for optimal model training and evaluation.
>
> 
🧠To evaluate the classification performance of different machine learning models (trained with 7–30 Hz filtered EEG data), we compare SVM, Random Forest, XGBoost, and KNN using several metrics:

| Model         | Accuracy | Precision (L/R)  | Recall (L/R)  | F1-score (L/R) |
|---------------|----------|------------------|---------------|----------------|
| **SVM**       | **0.89** | 0.91 / 0.87      | 0.87 / 0.92   | 0.89 / 0.90    |
| Random Forest | 0.84     | 0.87 / 0.82      | 0.81 / 0.88   | 0.84 / 0.85    |
| XGBoost       | **0.86** | 0.88 / 0.83      | 0.82 / 0.89   | 0.85 / 0.86    |
| KNN           | 0.74     | 0.81 / 0.69      | 0.62 / 0.86   | 0.71 / 0.77    |


## References

- Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). *BCI Competition 2008–Graz data set A*. Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology, 16(1-6), 1.

- Yang, L., Song, Y., Ma, K., & Xie, L. (2021). Motor imagery EEG decoding method based on a discriminative feature learning strategy. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 29, 368–379. 

- Gaur, P., Gupta, H., Chowdhury, A., McCreadie, K., Pachori, R. B., & Wang, H. (2021). A Sliding Window Common Spatial Pattern for Enhancing Motor Imagery Classification in EEG-BCI. *IEEE Transactions on Instrumentation and Measurement*, 70, 1–9. 

- Tibrewal, N., Leeuwis, N., & Alimardani, M. (2022). Classification of motor imagery EEG using deep learning increases performance in inefficient BCI users. *PLOS ONE*, 17(7), e0268880. 
