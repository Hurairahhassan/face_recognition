# import cv2 as cv
# import numpy as np
# import os
# from datetime import datetime
# import pandas as pd
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from sklearn.preprocessing import LabelEncoder
# import pickle
# from keras_facenet import FaceNet

# # Initialize FaceNet model
# facenet = FaceNet()

# # Load pre-trained model and label encoder
# faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
# Y = faces_embeddings['arr_1']
# encoder = LabelEncoder()
# encoder.fit(Y)

# # Load SVM model for face recognition
# model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# # Load Haar cascade for face detection
# haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Confidence threshold for face recognition
# CONFIDENCE_THRESHOLD = 0.8

# # Initialize video capture
# cap = cv.VideoCapture('rtsp://admin:admin123@192.168.1.104:554/cam/realmonitor?channel=1&subtype=1')
# if not cap.isOpened():
#     print("Error: Unable to open camera.")
#     exit()

# # Initialize attendance dataframe
# attendance = pd.DataFrame(columns=['Name', 'Date', 'Time'])

# # Initialize set to store recognized names
# recognized_names = set()

# while cap.isOpened():
#     _, frame = cap.read()
#     rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
#     for x, y, w, h in faces:
#         img = rgb_img[y:y+h, x:x+w]
#         img = cv.resize(img, (160, 160))
#         img = np.expand_dims(img, axis=0)
#         ypred = facenet.embeddings(img)
#         confidence = model.decision_function(ypred)  # Get confidence score from the model
#         if confidence.max() > CONFIDENCE_THRESHOLD:  # Correct the condition
#             face_name = model.predict(ypred)
#             final_name = encoder.inverse_transform(face_name)[0]
#             cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
#             cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
#                        1, (0, 0, 255), 3, cv.LINE_AA)
#             # Record attendance if the name is not already recorded in this session
#             if final_name not in recognized_names:
#                 now = datetime.now()
#                 date_time = now.strftime("%Y-%m-%d %H:%M:%S")
#                 attendance = pd.concat([attendance, pd.DataFrame({'Name': [final_name], 'Date': [now.date()], 'Time': [now.time()]})])
#                 recognized_names.add(final_name)
#     attendance.to_csv('attendance.csv', index=False)
#     cv.imshow("Face Recognition:", frame)
#     if cv.waitKey(1) & ord('q') == 27:
#         break

# cap.release()
# cv.destroyAllWindows()


import cv2 as cv
import numpy as np
import os
from datetime import datetime
import pandas as pd
import threading
import queue
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# Initialize FaceNet model
facenet = FaceNet()

# Load pre-trained model and label encoder
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

# Load SVM model for face recognition
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Load Haar cascade for face detection
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Confidence threshold for face recognition
CONFIDENCE_THRESHOLD = 0.8

# Initialize video capture
cap = cv.VideoCapture('rtsp://admin:admin123@192.168.1.104:554/cam/realmonitor?channel=1&subtype=1')
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Optimize camera settings for low-light
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 15)

# Initialize attendance dataframe
attendance = pd.DataFrame(columns=['Name', 'Date', 'Time'])

# Initialize set to store recognized names
recognized_names = set()

# Queue for frames
frame_queue = queue.Queue(maxsize=10)

def capture_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        # Skip frames to reduce load (capture every 3rd frame)
        if frame_queue.qsize() < 10:
            frame_queue.put(frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def process_frames():
    global attendance
    global recognized_names
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
            for x, y, w, h in faces:
                img = rgb_img[y:y+h, x:x+w]
                img = cv.resize(img, (160, 160))
                img = np.expand_dims(img, axis=0)
                ypred = facenet.embeddings(img)
                confidence = model.decision_function(ypred)
                if confidence.max() > CONFIDENCE_THRESHOLD:
                    face_name = model.predict(ypred)
                    final_name = encoder.inverse_transform(face_name)[0]
                    cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv.LINE_AA)
                    if final_name not in recognized_names:
                        now = datetime.now()
                        attendance = pd.concat([attendance, pd.DataFrame({'Name': [final_name], 'Date': [now.date()], 'Time': [now.time()]})])
                        recognized_names.add(final_name)
            attendance.to_csv('attendance.csv', index=False)
            cv.imshow("Face Recognition:", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

# Start capture and processing threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
capture_thread.start()
process_thread.start()

capture_thread.join()
process_thread.join()

cap.release()
cv.destroyAllWindows()

