# PPG-based BP Prediction ##
### CS 229 Final Project ###
#### By Rohan Sanda (rsanda@stanford.edu) and Edward Kim (edkim36@stanford.edu) ####

#### Description ####
This project takes a comprehensive approach to analyzing the BP prediction problem for future smart-watch applications. We begin by comparing three featurization techniques proposed in the literature: 1) hand-crafted physiological features (used by [2], [3], [4]) linear autoencoder embeddings, and 3) the processed PPG pulse (used by [5]). We find that the processed PPG pulse (denoted "raw signal") performs the best on our baseline models, achieving a best SBP MAE of 11.8 and DBP MAE of 7.0 using SVR Regression. Next, we compare three different neural network architecture and find that all improve our results significantly. We find that out of our ANN, CNN, and ResNet models, the CNN performs the best – achieving a SBP MAE of 10.1 and DBP MAE of 6.3. Moreoever, our models are relatively small and efficient compared to those found in literature. These results match existing literature on the topic [4], [5].

#### File Architecture ####
Construct the following architecture by creating and adding data to the `data` folder (processed using the `processor.ipynb` notebook). Update paths in `config.py`.
```
models
  ├── config.py
  ├── data.py
  ├── model.py
  ├── predict.py
  ├── run.py
  ├── trainer.py
  ├── autoencoder
      ├── linear_autoencoder.py
      └── lstm_autoencoder.py
  └── feature_extraction
      └── feat_extract.ipynb
processing
    ├── processor.ipynb
    └── post_processing_analysis.ipynb
data
    ├── segments.pickle
    └── bps.pickle
```

#### Autoencoders and Feature Extraction ####
To experiment with using featurized versus raw signal data in our baseline models, check out the `autoencoder` and `feature_extraction` directories. Results can be found in the final report. It is recommended to use the raw signal data for the neural network models (1D-CNN, ResNet+1D-CNN, and ANN).   

#### Running the Model ####
Set parameters in `config.py`. Then run `run.py` --> `python3 run.py`. Raw data can be processed using the `processor.ipynb` script. Raw data was obtained from the VitalDB dataset [1]. Please read our final report for more details. 

##### Citations #####
[1] HC. Lee, Y. Park, and S.B. Yoon. Vitaldb, a high-fidelity multi-parameter vital signs database in surgical patients. Nature Scientific Data, 9(279), 2022.
[2] Seungman Yang, Jangjay Sohn, Saram Lee, Joonnyong Lee, and Hee Chan Kim. Estimation and validation of arterial blood pressure using photoplethysmogram morphology features in conjunction with pulse arrival time in large open databases. IEEE Journal of Biomedical and Health Informatics, 25(4):1018–1030, 2021.
[3] Clémentine Aguet, Jérôme Van Zaen, João Jorge, Martin Proença, Guillaume Bonnier, Pascal Frossard, and Mathieu Lemay. Feature learning for blood pressure estimation from photoplethysmography. In 2021 43rd Annual International Conference of the IEEE Engineering in Medicine Biology Society (EMBC), pages 463–466, 2021.
[4] Umapathy Mangalanathan V. Jeya Maria Jose M. Anand Geerthy Thambiraj, Uma Gandhi. Investigation of the effect of womerseley number, ecg and ppg features for cuff-less blood pressure estimation using machine learning. Biomedical Signal Processing and Control, 60, 2020.
[5] Nejc Mlakar Slapnicˇar, Gašper and Mitja Luštrek. Blood pressure estimation from photoplethysmogram using a spectro-temporal deep neural network. Sensors, 9(15), 2019.



 


