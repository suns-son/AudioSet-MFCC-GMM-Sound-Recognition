import os
import shutil
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yt_dlp
import matplotlib.pyplot as plt

# 1. Dataset Klasöründen Test Dataset Klasörüne Dosya Kopyalama

def extract_labels(filename):
    start = filename.find('.')
    end = filename.rfind('.')
    if start != -1 and end != -1 and start < end:
        label_part = filename[start:end]
        label_part = label_part.replace('.', '/')
        labels = label_part.split('-')
        return labels
    return []

target_labels = [
    "/m/012ndj", "/m/081rb", "/m/0dl83", "/m/07rpkh9", "/m/03fwl",
    "/m/02zsn", "/m/01w250", "/m/05tny_", "/m/05zppz", "/m/0d31p"
]

dataset_dir = 'dataset'
test_dataset_dir = 'test_dataset'

if not os.path.exists(test_dataset_dir):
    os.makedirs(test_dataset_dir)

for filename in os.listdir(dataset_dir):
    labels = extract_labels(filename)
    if any(label in target_labels for label in labels):
        shutil.copy(os.path.join(dataset_dir, filename), test_dataset_dir)

# 2. CNN Modeli ile Eğitim

def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]
    else:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def load_data(data_dir):
    X = []
    y = []
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith(".mp3"):
            labels = extract_labels(filename)
            if labels:
                mfcc = extract_mfcc(os.path.join(data_dir, filename))
                X.append(mfcc)
                y.append(labels[0])  # İlk label'ı alıyoruz
    return np.array(X), np.array(y)

X, y = load_data(test_dataset_dir)

X = X[..., np.newaxis]  # CNN için kanal boyutu ekliyoruz

le = LabelEncoder()
y = le.fit_transform(y)
y = tf.keras.utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 3. YouTube Videolarını İndirme

yt_urls = [
    "https://www.youtube.com/watch?v=iu_1UeTGigc",
    "https://www.youtube.com/watch?v=9m6yx8BpAZA",
    "https://www.youtube.com/watch?v=dBH6LSnOoPk",
    "https://www.youtube.com/watch?v=eGwnsdRzDv8",
    "https://www.youtube.com/watch?v=1r4ArWJugvY",
    "https://www.youtube.com/watch?v=CMrHF9p5sII",
    "https://www.youtube.com/watch?v=AxP_AFB-Tw4",
    "https://www.youtube.com/watch?v=ThVluC7YRDs",
    "https://www.youtube.com/watch?v=GJ99q_MJPJA",
    "https://www.youtube.com/watch?v=xysIN10DgL8"
]

def download_audio(url, output_dir):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s')
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

for url in yt_urls:
    download_audio(url, test_dataset_dir)

# 4. Modeli Test Etme ve Tutarlılık Kontrolü

def evaluate_model(model, test_dir, le):
    X_test_new = []
    y_test_new = []
    for filename in os.listdir(test_dir):
        if filename.endswith(".mp3"):
            labels = extract_labels(filename)
            if labels:
                mfcc = extract_mfcc(os.path.join(test_dir, filename))
                X_test_new.append(mfcc)
                y_test_new.append(labels[0])
    
    X_test_new = np.array(X_test_new)[..., np.newaxis]
    y_test_new = le.transform(y_test_new)
    y_test_new = tf.keras.utils.to_categorical(y_test_new)
    
    loss, accuracy = model.evaluate(X_test_new, y_test_new)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    
    return accuracy

accuracy = evaluate_model(model, test_dataset_dir, le)

# 5. Görselleştirme

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')
plt.show()
