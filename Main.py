import requests
import pyaudio
import wave
from keras import models
import cv2 as cv
import numpy as np

import Real_Model.Create_Spectogram as cs # See my other repository that you can use to make spectogram


def predict():
    class_names = ["ClashOfClans", "Nothing"]

    model = models.load_model("Real_Model/ClashOfClans_Nothing_Cough_1.h5")

    img = cv.imread("out2/output.wav.png")
    img = cv.resize(img, (64, 64))

    i = np.array([img]) / 255
    prediction = model.predict(i)
    index = np.argmax(prediction)

    print("Prediction : ", class_names[index])
    print("pred: ", prediction)

    return class_names[index]


def record_audio(time):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = time
    filename = "out/output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')
    print('For ', time, "secs")

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


def get_user():
    response = requests.get(
        "https://api.clashofclans.com/v1/players/%23LYY22Y29Y", headers=headers)
    user_json = response.json()
    print(user_json)


if __name__ == '__main__':
    headers = {
        "Accept": "application/json",
        "authorization": "Bearer YourApiKey"
    }

    i = input("Record / Save / Action ? ")
    if i == "Record":
        record_audio(5)  # record 5 sec audio
    elif i == "Save":
        cs.main()  # save the 5 sec audio as a spectogram
    elif i == "Action":
        prediction = predict()  # predict if clash of clans or nothing

        if prediction == "ClashOfClans":
            get_user()  # get the data of plaayer from clash of clans api
        else:
            print("Gesundheit !")

    else:
        print("Unbekannt")
