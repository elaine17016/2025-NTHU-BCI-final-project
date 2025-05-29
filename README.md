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
- **Dataset Page**: [https://www.bbci.de/competition/iv/#dataset2a](https://www.bbci.de/competition/iv/#dataset2a)
- **Owner / Source**: BCI Group, Department of Computer Science, Technische Universität Berlin, Germany.
- **Citation**:
  > Tangermann, M., Müller, K. R., Aertsen, A., Birbaumer, N., Braun, C., Brunner, C., ... & Blankertz, B. (2012). Review of the BCI Competition IV. Frontiers in Neuroscience, 6, 55. [DOI: 10.3389/fnins.2012.00055](https://doi.org/10.3389/fnins.2012.00055)






# 🧠 Why we choose 7-30Hz? Comparison of EEG Frequency Bands on Machine Learning Performance
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

##  Conclusion & Recommendation

- The **7–30 Hz** band performs best, with both validation and test accuracy reaching **91%**.
- The **1–50 Hz** range includes high-frequency noise, slightly degrading validation performance.
- The **7–13 Hz** band focuses on alpha rhythms, which may not fully capture the motor imagery signals, resulting in slightly lower accuracy.

> We recommend using the **7–30 Hz** frequency band for optimal model training and evaluation.
> 

## References

- Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). *BCI Competition 2008–Graz data set A*. Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology, 16(1-6), 1.

- Yang, L., Song, Y., Ma, K., & Xie, L. (2021). Motor imagery EEG decoding method based on a discriminative feature learning strategy. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 29, 368–379. https://doi.org/10.1109/TNSRE.2021.3058861

- Gaur, P., Gupta, H., Chowdhury, A., McCreadie, K., Pachori, R. B., & Wang, H. (2021). A Sliding Window Common Spatial Pattern for Enhancing Motor Imagery Classification in EEG-BCI. *IEEE Transactions on Instrumentation and Measurement*, 70, 1–9. https://doi.org/10.1109/TIM.2021.3061167

- Tibrewal, N., Leeuwis, N., & Alimardani, M. (2022). Classification of motor imagery EEG using deep learning increases performance in inefficient BCI users. *PLOS ONE*, 17(7), e0268880. https://doi.org/10.1371/journal.pone.0268880
