# EEG Classification Model

## 📌 Project Overview
This project focuses on analyzing and classifying Electroencephalogram (EEG) signals to differentiate between various brain states. By leveraging advanced preprocessing techniques and machine learning models, the goal is to enhance the accuracy of EEG signal classification, contributing to improved neurological assessments.

## 🚀 Features
- **Data Preprocessing**: Cleaning and preparing EEG data for analysis.
- **Feature Extraction**: Deriving meaningful features from raw EEG signals.
- **Model Implementation**: Utilizing machine learning models, such as Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs), for classification tasks.
- **Performance Evaluation**: Assessing model accuracy and other relevant metrics.

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn
- **Tools**: Jupyter Notebook


## 📊 Dataset
- **Source**: [Bonn University EEG Dataset](https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/)
- **Description**: The dataset comprises EEG recordings categorized into five sets (A-E), each containing 100 single-channel EEG segments of 23.6 seconds duration. Sets A and B were recorded from healthy volunteers with eyes open and closed, respectively. Sets C, D, and E were recorded from epileptic patients during seizure-free intervals and seizure activity.

## 🔍 Methodology
1. **Data Preprocessing**:
   - **Filtering**: Applied bandpass filters to remove noise and irrelevant frequencies.
   - **Normalization**: Scaled data to ensure uniformity across features.
   - **Segmentation**: Divided continuous EEG signals into fixed-length segments for analysis.

2. **Feature Extraction**:
   - **Time-Domain Features**: Extracted statistical measures such as mean, variance, skewness, and kurtosis.
   - **Frequency-Domain Features**: Computed power spectral densities using Fast Fourier Transform (FFT).
   - **Time-Frequency Analysis**: Applied wavelet transforms to capture transient features.

3. **Model Implementation**:
   - **Convolutional Neural Network (CNN)**: Designed to capture spatial features from EEG signals.
   - **Long Short-Term Memory (LSTM)**: Utilized to learn temporal dependencies in sequential data.
   - **Hybrid CNN-LSTM**: Combined spatial and temporal feature learning for improved performance.

4. **Performance Evaluation**:
   - **Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix.
   - **Cross-Validation**: Employed k-fold cross-validation to ensure model robustness.

## 📈 Results
- **Accuracy**: Achieved an accuracy of 98% on the test set using the hybrid CNN-LSTM model.
- **Confusion Matrix**: Displayed high true positive rates across all classes, indicating reliable classification performance.


