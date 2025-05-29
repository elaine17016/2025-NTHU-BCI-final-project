# 2025-NTHU-BCI-final-project

## Group 4  
113062524 林玟萱  
113062608 王彥翔  
110030034 鄭羿羚

## 1. Introduction

BCI (Brain-Computer Interface), this technology holds great potential for assisting individuals with physical disabilities, particularly through the use of MI (Motor Imagery). While MI-BCI systems often require extensive user training, we explore whether DL (Deep Learning) can improve signal recognition, especially for inefficient users, thereby enhancing the overall effectiveness and accessibility of MI-based BCI applications.

## 2. Data Collection

**Dataset:** The BCI Competition IV – Dataset 2a  
**Subjects:** 9 subjects  
Each subject underwent two experiments, each consisting of 6 runs, for a total of 288 trials with 48 trials per run.  

**Imagery tasks:**  
- Left hand  
- Right hand  
- Both feet  
- Tongue  

**Trial timing:**  
- t = 0 s: Fixation and beep  
- t = 2 s: Arrow cue (1.25 s)  
- t = 2 to 6 s: Motor imagery  
- t = 6 to 7.5 s: Rest
  
 ![Timing Scheme Paradigm](https://raw.githubusercontent.com/orvindemsy/BCICIV2a-FBCSP/d6dce55b4951b8e46bb5e625b060d332101cdd59/img/timing-scheme-paradigm.png)


## References

- Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). *BCI Competition 2008–Graz data set A*. Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology, 16(1-6), 1.

- Yang, L., Song, Y., Ma, K., & Xie, L. (2021). Motor imagery EEG decoding method based on a discriminative feature learning strategy. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 29, 368–379. https://doi.org/10.1109/TNSRE.2021.3058861

- Gaur, P., Gupta, H., Chowdhury, A., McCreadie, K., Pachori, R. B., & Wang, H. (2021). A Sliding Window Common Spatial Pattern for Enhancing Motor Imagery Classification in EEG-BCI. *IEEE Transactions on Instrumentation and Measurement*, 70, 1–9. https://doi.org/10.1109/TIM.2021.3061167

- Tibrewal, N., Leeuwis, N., & Alimardani, M. (2022). Classification of motor imagery EEG using deep learning increases performance in inefficient BCI users. *PLOS ONE*, 17(7), e0268880. https://doi.org/10.1371/journal.pone.0268880
