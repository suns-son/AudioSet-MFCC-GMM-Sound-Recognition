# AudioSet MFCC GMM Sound Recognition

An advanced sound recognition project leveraging AudioSet, featuring custom dataset creation, MFCC feature extraction, and a convolutional neural network (CNN) for sound classification. This repository includes tools for dataset preparation, translation of labels, model training, and performance evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

This project focuses on the development of a robust sound recognition system using the AudioSet dataset. The primary objective is to create a comprehensive dataset, extract MFCC features, and train a convolutional neural network (CNN) to accurately classify various sounds.

## Dataset Preparation

### Downloader Module

The `downloader.py` module utilizes `yt-dlp` and `ffmpeg` libraries to download and convert YouTube videos into audio files. The `balanced_train_segments.csv` file, which contains 10-second segments of YouTube videos, is used to create the dataset. Despite challenges like deleted or private videos, approximately 83% of the dataset was successfully downloaded and converted to MP3 format. The filenames are structured to include labels, eliminating the need for a separate label file.

### Translator Module

The `translator.py` module translates the labels from the `ontology.json` file into Turkish using the `deep_translator` library. The translations are saved in a `translated_ontology.json` file. The `tqdm` library is used to display progress during the translation process.

### Info Module

The `info.py` module categorizes and analyzes the downloaded sounds based on their labels. It displays the label IDs, their English and Turkish names, and the count of sounds for each label. A subset of 10 labels with 50 audio samples each was selected for training, ensuring a balanced and comprehensive test dataset.

Note: While `balanced_train_segments.csv` and `ontology.json` are used in the code, you do not need to use the versions included in this repository. They will be automatically downloaded during the dataset preparation process.

## Model Training

The `test.py` module handles the training of a CNN model using the prepared dataset. Key steps include:

- **MFCC Feature Extraction:** Using `librosa`, MFCC features are extracted from the audio files.
- **Label Encoding:** Labels are encoded into numerical values using `LabelEncoder`.
- **Model Architecture:** A CNN model is defined with layers for convolution, pooling, flattening, dense connections, and dropout.
- **Training:** The model is trained for 20 epochs using `tensorflow` and `keras` libraries, with the training history recorded.

## Model Evaluation

The trained model is evaluated using additional YouTube videos. The `test.py` module includes functionality to download these videos, extract MFCC features, and assess the model's performance. Accuracy and other metrics are computed and visualized.

## Dependencies

- `requests`
- `yt-dlp`
- `ffmpeg`
- `deep_translator`
- `tqdm`
- `numpy`
- `librosa`
- `tensorflow`
- `scikit-learn`
- `matplotlib`

## Usage

1. **Clone the Repository:**
   """
   git clone https://github.com/suns-son/AudioSet-MFCC-GMM-Sound-Recognition.git
   cd AudioSet-MFCC-GMM-Sound-Recognition
   """

2. **Install Dependencies:**
   """
   pip install -r requirements.txt
   """

3. **Prepare the Dataset:**
   - Download the dataset from this [Google Drive link](https://drive.google.com/drive/folders/1O3xPaE53pAH5-v_sbHYlxn5YNDoZ2kQq).
   - Place the 18291 audio files into a folder named `dataset` in the same directory as the code files.

4. **Run Dataset Preparation:**
   """
   python downloader.py
   python translator.py
   python info.py
   """

5. **Train the Model:**
   """
   python test.py
   """

## Results

The model achieved an accuracy of approximately 85% during training. However, due to the small dataset, overfitting was observed. Further data augmentation and regularization techniques can be explored to improve model generalization.
