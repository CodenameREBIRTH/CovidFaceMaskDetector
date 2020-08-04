# importing necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import cv2 as cv
import numpy as np
import time
import os
import imutils
import argparse


def detect_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    # crete 4D array blob
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detection = faceNet.forward()

    # list of ROI(faces) , face locations , predictions
    faces = []
    locations = []
    predictions = []

    for n in range(0, detection.shape[2]):
        confidence = detection[0, 0, n, 2]
        if confidence > 0.5:
            # compute x y coordinate of bounding box
            box = detection[0, 0, n, 3:7] * np.array([w, h, w, h])
            (x_start, y_start, x_end, y_end) = box.astype("int")
            # bounding box fall within range of frame
            (x_start, y_start) = (max(0, x_start), max(0, y_start))
            (x_end, y_end) = (min(w - 1, x_end), min(h - 1, y_end))
            # get the ROI for face image and perform other operations
            face = frame[y_start:y_end, x_start:x_end]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locations.append((x_start, y_start, x_end, y_end))
    # make prediction only when face is detected
    if len(faces) > 0:
        # for faster computation make batch prediction on all faces detected at the same time rather than one by one
        # prediction
        predictions = maskNet.predict(faces)

    return locations, predictions


# load serialized face mask detector model from disk
print("loading face mask detector model")
prototxtPath = os.path.sep.join(["FaceDetector", "deploy.prototxt"])
weigthPath = os.path.sep.join(["FaceDetector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv.dnn.readNet(prototxtPath, weigthPath)

# load face mask detector model from disk
maskNet = load_model("FaceMaskDetector")

# initialise video stream
print("starting camera")
cap = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    # get camera frame and resize it to 400 pixels width
    frame = cap.read()
    frame = imutils.resize(frame, width=400)

    (locations, predictions) = detect_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locations, predictions):
        (x_start, y_start, x_end, y_end) = box
        (yesMask, noMask) = pred

        # determine class label, probability and color bounding boxes
        label = "With Mask" if yesMask > noMask else "Without Mask"
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(yesMask, noMask) * 100)

        cv.putText(frame, label, (x_start, y_start - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)

    # show output and break loop when 'q' is pressed
    cv.imshow("Be Safe And Be Healthy", frame)
    if cv.waitKey(10) == ord('q'):
        break
# destroy all windows
cv.destroyAllWindows()
cap.stop()
