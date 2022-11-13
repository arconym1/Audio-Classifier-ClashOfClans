import os
import random
import numpy as np
from keras import models
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import cv2


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def normalize(from_path, dest):
    for file in os.listdir(from_path):
        sound = AudioSegment.from_file(from_path + file, "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export(f"{dest + file}.wav", format="wav")
        print("Sucessfully exported : ", file)


def create_training_data(Categories, Data_dir, training_data):
    for category in Categories:
        path = os.path.join(Data_dir, category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))  # Gray Scale = 2D, else 3D
                new_array = cv2.resize(img_array, (64, 64))
                training_data.append([new_array, class_num])

            except Exception as e:
                print(e)


def train():
    Data_dir = "/WordClassify_Spectogram-CNN/Real_Model/data/Spectograms/"
    Categories = ["ClashOfClans", "Nothing"]
    training_data = []

    create_training_data(Categories, Data_dir, training_data)
    random.shuffle(training_data)
    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X = X / 255.0
    print(X.shape)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))  # <- 6. = 128, 128, 3
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(2, activation="softmax"))  # Or sigmoid

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=10)

    model.save(
        "ClashOfClans_Nothing_Cough_1.h5")


if __name__ == '__main__':
    # normalize("data/Sounds/ClashOfClans/", "data/Normalized_Sounds/ClashOfClans_Normalized/")
    # normalize("data/Sounds/Nothing/", "data/Normalized_Sounds/Nothing_Normalized/")
    train()
