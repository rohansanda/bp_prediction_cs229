# PPG-based BP Prediction ##
### CS 229 Final Project ###
#### By Rohan Sanda and Edward Kim ####

### File Architecture
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

### Running the Model
Set parameters in `config.py`. Then run `run.py` --> `python3 run.py`. Raw data can be processed using the `processor.ipynb` script. Raw data was obtained from the VitalDB dataset [1]. Please read our final report for more details. 

[1] HC. Lee, Y. Park, and S.B. Yoon. Vitaldb, a high-fidelity multi-parameter vital signs database in surgical patients. Nature Scientific Data, 9(279), 2022.

 


