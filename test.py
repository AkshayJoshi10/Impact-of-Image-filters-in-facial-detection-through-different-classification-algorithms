#!/usr/bin/env python
# coding: utf-8

# In[17]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video = cv2.VideoCapture(1)
facedetect = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

# with open('labels.txt', 'r') as f:
#     LABELS = f.read().splitlines()

# # LABELS = ['0', '1', '2', '3']  # Replace this with your actual list

# label_mapping = {'0': 'Aadershi', '1': 'Akshay', '2': 'Ayush', '3': 'Daivik', '4': 'Mayank', '5': 'Payal', '6': 'Yashi', '7': 'Yogesh'}

# # Map numerical labels to names using list comprehension
# LABELS = [label_mapping[label] for label in LABELS]
LABELS = ['Aadershi', 'Akshay', 'Ayush', 'Daivik', 'Mayank', 'Payal', 'Yashi', 'Yogesh']
model = load_model('best_model.h5')  # Load your trained CNN model

imgBackground = cv2.imread("../Attendance.png")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (880, 924))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        resized_img = cv2.resize(crop_img, (128, 128))

        resized_img = img_to_array(resized_img) / 255.0
        resized_img = np.expand_dims(resized_img, axis=0)
        output = model.predict(resized_img)
        predicted_label = LABELS[np.argmax(output)]

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, predicted_label, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        attendance = [predicted_label, str(timestamp)]

    imgBackground[0:0 + 1080, 0:0 + 880] = frame
    cv2.imshow("Frame", imgBackground)
    k = cv2.waitKey(1)

    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)

        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


# In[13]:


LABELS


# In[14]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video = cv2.VideoCapture(1)
facedetect = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

with open('labels.txt', 'r') as f:
    LABELS = f.read().splitlines()

# LABELS = ['0', '1', '2', '3']  # Replace this with your actual list

label_mapping = {'0': 'Aadershi', '1': 'Akshay', '2': 'Ayush', '3': 'Daivik', '4': 'Mayank', '5': 'Payal', '6': 'Yashi', '7': 'Yogesh'}

# Map numerical labels to names using list comprehension
LABELS = [label_mapping[label] for label in LABELS]
LABELS = ['Aadershi','Akshay','Ayush','Daivik','Mayank','Payal','Yashi','Yogesh']
model = load_model('best_model.h5')  # Load your trained CNN model

imgBackground = cv2.imread("../Attendance.png")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (880, 924))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Initialize variables to keep track of the largest face
    largest_face_area = 0
    largest_face = None

    for (x, y, w, h) in faces:
        # Calculate the area of the current face
        current_face_area = w * h

        # Update largest_face if the current face is larger
        if current_face_area > largest_face_area:
            largest_face_area = current_face_area
            largest_face = frame[y:y + h, x:x + w, :]

    if largest_face is not None:
        resized_img = cv2.resize(largest_face, (128, 128))

        resized_img = img_to_array(resized_img) / 255.0
        resized_img = np.expand_dims(resized_img, axis=0)
        output = model.predict(resized_img)
        predicted_label = LABELS[np.argmax(output)]
        print(predicted_label)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, predicted_label, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        attendance = [predicted_label, str(timestamp)]

    imgBackground[0:0 + 1080, 0:0 + 880] = frame
    cv2.imshow("Frame", imgBackground)
    k = cv2.waitKey(1)

    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)

        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()




