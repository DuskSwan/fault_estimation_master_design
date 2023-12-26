# My Master's Thesis Project

This is the project I created for my master's thesis. Although the main content of the thesis has been completed before, I hope to use this opportunity to experience the entire process of creating a project.

The goals I hope to achieve include:

- [x] Use a clear and versatile project structure template to manage scripts.
- [x] Use advanced deep learning management tools such as yacs and ignite.
- [x] Use the log management library loguru to properly record logs.

## Introduction

This project aims to identify faults and determine the severity of faults based on bearing vibration signals. The flowchart for fault identification is shown below:

![flowchart](resource/img/flowchart.png)

For detailed ideas, please refer to my [paper](resource/doc/paper.docx). I will also supplement the project introduction here.

## Code Framework

The template used in this project is from [here](https://github.com/DuskSwan/Deep-Learning-Project-Template). Although I preserved the README of the template, I will have my own understanding and adjustments in the implementation, so I will clarify them here.

The original template structure can be found in README_template.md. The modified structure by me is as follows:

```text
├──  config
│    └── __init__.py
│    └── default.py
|       Common parameter configurations, the class cfg in defaults.py is the final one used.
│    └── data_usage.yml
|       Configuration options that need to be modified in the experiment.
│ 
│
├──  data
│    └── __init__.py
│    └── datasets
|       This directory stores datasets, ideally without other scripts.
│    └── build.py
|       Provide dataloader and mini-batch.
│
│
├──  engine
│    ├── trainer.py     - this file contains the train loops.
│    └── inference.py   - this file contains the inference process.
|       Provide functions for training and testing.
│
│
├──  modeling
│    └── __init__.py
│    └── LSTM.py
|       Provide models and customized layers.
│
├──  solver
│    └── __init__.py
│    └── build.py
|       Customize solvers, including optimizers and learning rate schedulers.
│ 
├──  run   - Scripts that will actually run are placed here.
│    └── train_net.py  
│    └── test.py
│    └── tools.py
|       tools contains high-level functions used for training and testing	
|       
│ 
└──  utils
│    ├── __init__.py
│    └── feature.py
│    └── similarity.py
│    └── threshold.py
|       Utility tools. In this project, tools are needed for feature extraction, similarity calculation, threshold calculation, etc.
│ 
└──  log    - Stores logs
|    └── CWRU_train_2023-12-19-20-06-35.log
│ 
└──  output    - Stores output content, such as trained models
     └── cwru_ltsm.pth
         
```

## Detailed Explanation of Approach

This algorithm aims to provide a fault diagnosis method based on engine vibration signals. According to the idea, fault diagnosis (determining whether there is a fault), fault detection (determining the time point when the fault occurs), and fault severity evaluation are all possible, but currently only fault diagnosis is implemented.

The core idea of this method is to use a time series prediction model to predict the feature sequences extracted from normal signals. Signals that are consistent with the prediction are considered normal, while those with a large difference from the predicted value indicate a fault, and the larger the deviation, the more severe the fault. It can be divided into the following steps:

### 1. Using sliding windows to convert signals into multiple samples

Let the original signal be represented as $\{x_1,x_2,...,x_N\}$, where $x_i$ is a vector of dimension channel. It is represented as a 2D array of size N x channel in the program. It needs to be converted into multiple signal segments, with each segment calculating a feature vector.

A sliding window of length sublen is used to sequentially extract from the original signal. To obtain as many samples as possible, the window's moving step is set to 1. In this way, up to N-sublen+1 signal segments can be taken from the original signal, and the number of segments to be taken is set to piece. Each signal segment has a length of sublen. The kth sample (signal segment) is represented as $\{x_k,x_{k+1},...,x_{k+sublen-1}\}$, which is a sublen x channel 2D array in the program. All signal segments form a piece x sublen x channel 3D array.

### 2. Feature extraction and selection

18 available features are predefined. See the definition in the [paper](resource/doc/paper.docx). It should be noted that the feature calculation formula mentioned in the paper is only applicable to one-dimensional signals. However, real signal acquisition involves multiple measurement points, and each measurement point's data constitutes a signal channel. Therefore, in practical processing, features should be individually extracted for each channel of the signal. The final feature time series has a dimension equal to the product of the number of original signal channels and the number of features.

For the input signal, after converting it into multiple signal segments using sliding windows, each channel of each signal segment can calculate channel x 18 features, which means each signal segment matrix generates a feature vector of dimension channel x 18. Therefore, each original signal (N rows, channel columns) is converted into a feature time series (piece rows, channel x 18 columns).

If we use the 18-dimensional time series as the prediction object, the model will be difficult to converge. Therefore, we hope to select a subset of features. Since the goal is to reflect the existence of faults through prediction accuracy, we should expect a significant difference between the feature time series of normal signals and faulty signals. Therefore, for each feature extracted from normal and faulty signals, calculate the similarity of the corresponding features one by one, and prioritize the use of features with low similarity.

In the selection of similarity, the Dynamic Time Warping (DTW) algorithm is used to calculate the DTW scores of two feature matrices column by column. It is worth noting that for a certain feature (such as Mean), since a sequence will be calculated on each channel (assuming there are three channels, there exist chn1_Mean, chn2_Mean, chn3_Mean), each channel has a corresponding DTW score. We take the sum of the scores of each channel as the final score of this feature.

Sort the features in descending order according to their scores to obtain the order of feature selection. Generally, selecting the first feature is sufficient to distinguish between normal and faulty signals.

### 3. Establishing Prediction Model

Assuming that the selected feature is only Mean. Next, a feature sequence sample is constructed for training the model.

Applying the sliding window again, the feature matrix of the normal signal is used to extract the feature matrix sample sequence for training the LSTM prediction model. The LSTM model aims to predict the values of the next p points (feature vectors) based on the preceding m points (feature vectors) in the sequence. Therefore, the sliding window length is set as fsublen = m + p. The sliding stride is still 1. The training dataset should contain at least fpiece samples, so the length of the feature time series, piece, must be greater than or equal to fsublen + fpiece - 1.

Each feature sequence sample is a matrix with fsublen rows and channel columns (if two features are selected, it will be channel × 2 columns). The first m rows and channel columns are used as the input of the sample, and the last p rows and channel columns are used as the output. Using fpiece samples, train the time series prediction model. Here, I am using a simple network consisting of an LSTM layer and a fully connected layer.

### 4. Computing the Discrimination Threshold

Given a well-trained model, for any segment of the original signal, after sliding window, a signal sample set is obtained and feature extraction is performed to obtain the feature time series. Once again, sliding window is performed to obtain the feature time series samples. These samples are input into the model to calculate the error scores between the real values and the predicted values. We use the Mean Absolute Error (MAE) as the measurement metric for the error scores.

For each sample, both the predicted values and the real values are arrays with p rows and channel columns, from which an MAE is calculated. By extracting a series of feature sequence samples from the normal signal and inputting them into the prediction model, multiple MAE values are obtained. They all belong to the "normal signal MAE distribution". For an unknown signal, if the calculated MAE belongs to the "normal signal MAE distribution", then the signal is considered normal; otherwise, it is a fault signal. Moreover, the MAE of a fault signal is always larger than that of a normal signal.

As a result, the fault detection is transformed into a hypothesis testing problem. Assuming the null hypothesis H0: the MAE of the unknown signal belongs to the "normal signal MAE distribution", if there is enough evidence to reject the null hypothesis, it is determined that the unknown signal is a fault signal; otherwise, it is normal. We hope to find a suitable statistical measure as the threshold, and when the MAE of the unknown signal is greater than this threshold, the null hypothesis is rejected.

For a normal distribution, μ+3σ is commonly used as the standard to test for outlier values. However, experiments have shown that the MAE of the normal signal does not follow a normal distribution. For skewed distributions, the following strategies can be used: 1. Taking the logarithm to transform the data into a normal distribution; 2. Using the modified z-score based on the Interquartile Range (IQR); 3. Using the p-quantile; 4. Using the box plot.

Through experimental verification, the modified z-score based on the IQR performs the best. Its calculation formula is Z = (x - x_M) / (IQR / 1.349), where x_M is the median of the known data and IQR is the difference between the third quartile (Q3) and the first quartile (Q1) of the known data. If the score Z calculated for data point x is greater than 3.5, x is considered an outlier. Thus, the threshold λ is calculated as λ = x_M + (3.5 * IQR) / 1.349. If x > λ, x is considered an outlier.

### 5. Predicting Unknown Signals

For a segment of unknown signal, after sliding window, a signal sample set is obtained and feature extraction is performed to obtain the feature time series. Once again, sliding window is performed to obtain the feature time series samples. These samples are input into the model to calculate the MAE of the predicted values. Each sample will have an MAE value, and the average is taken as the measurement metric. If this metric is greater than the threshold, the unknown signal is determined to be a fault signal; otherwise, it is normal.
