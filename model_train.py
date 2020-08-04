# importing necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from imutils import paths

# initialise hyperparameters using learning rate decay schedule
learning_rate = 1e-4
epochs = 20
batch_size = 32

print("loading images")
path = list(paths.list_images("Dataset"))
data = []
label = []

for imagePath in path:
    # extract class label from filename
    classLabel = imagePath.split(os.path.sep)[-2]
    # load image and preprocess it
    img = load_img(imagePath, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    # update data and label list
    data.append(img)
    label.append(classLabel)
# convert data and label list to np array
data = np.array(data, dtype="float32")
label = np.array(label)

# perform encoding on label array
lb = LabelBinarizer()
label = lb.fit_transform(label)
label = to_categorical(label)

# partition dataset into train test set
(x_train, x_test, y_train, y_test) = train_test_split(
    data,
    label,
    test_size=0.20,
    stratify=label,
    random_state=42)

# image generator for data augmentation
data_aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.20,
    height_shift_range=0.20,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# load MobileNetV2 model without top layer and then adding modified head
base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))
head = base.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten(name="flatten")(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(2, activation="softmax")(head)

model = Model(inputs=base.input, outputs=head)

# freeze all layers in base model for transfer learning
for layer in base.layers:
    layer.trainable = False

# compile model
print("compiling model")
optm = Adam(lr=learning_rate, decay=learning_rate/epochs)
model.compile(
    loss="binary_crossentropy",
    optimizer=optm,
    metrics=["accuracy"])

# train top layer of model created above
print("training top layer")
top = model.fit(
    data_aug.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train)//batch_size,
    validation_data=(x_test, y_test),
    validation_steps=len(x_test)//batch_size,
    epochs=epochs
)

# predictions on testing set
print("evaluate model")
prediction = model.predict(x_test, batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)
print(classification_report(y_test.argmax(axis=1), prediction, target_names=lb.classes_))

# saving model
print("saving model")
model.save("FaceMaskDetector", save_format="h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), top.history["loss"], label="training_loss")
plt.plot(np.arange(0, epochs), top.history["val_loss"], label="validation_loss")
plt.plot(np.arange(0, epochs), top.history["accuracy"], label="training_accuracy")
plt.plot(np.arange(0, epochs), top.history["val_accuracy"], label="validation_accuracy")
plt.title("error analysis")
plt.xlabel("epochs count")
plt.ylabel("loss/accuracy")
plt.legend(loc="upper right")
plt.savefig("DataPlot.png")