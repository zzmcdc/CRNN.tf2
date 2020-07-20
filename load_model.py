import os
from tensorflow import keras
from model import build_model
restore = './026_0.7094_0.3820.h5'

with open("./label.txt") as to_read:
    classes = list(to_read.read().strip())

num_classes = len(classes) + 1
model = build_model(num_classes, 256, 1)
model.load_weights(restore, by_name=True, skip_mismatch=True)
# model.summary()
model.save("crnn_tf")
