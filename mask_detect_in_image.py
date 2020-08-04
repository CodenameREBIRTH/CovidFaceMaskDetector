# importing necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import time
import os
from imutils import paths
import argparse

# load serialized face mask detector model from disk
print("loading face mask detector model")
prototxtPath = os.path.sep.join(["FaceDetector", "deploy.prototxt"])
weigthPath = os.path.sep.join(["FaceDetector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv.dnn.readNet(prototxtPath, weigthPath)

# load face mask detector model from disk
maskNet = load_model("FaceMaskDetector")

# load image from disk, clone it
print("loading image from disk")
path = list(paths.list_images("TestImage"))

for imagePath in path:
    img = cv.imread(imagePath)
    (h, w) = img.shape[:2]
    # crete 4D array blob
    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # compute face detections
    faceNet.setInput(blob)
    detection = faceNet.forward()

    # iterate over detections
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
            face = img[y_start:y_end, x_start:x_end]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # determine class label, probability and color bounding boxes
            (yesMask, noMask) = maskNet.predict(face)[0]
            label = "With Mask" if yesMask > noMask else "Without Mask"
            color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(yesMask, noMask) * 100)

            # display bounding box with label and color
            cv.putText(img, label, (x_start, y_start - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv.rectangle(img, (x_start, y_start), (x_end, y_end), color, 2)

    # show output and break loop when 'q' is pressed
    cv.imshow("Be Safe And Be Healthy", img)
    cv.waitKey(0)



