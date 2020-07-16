import os
from model import build_model
restore = '015_0.4931_0.3950.h5'

with open("./label.txt") as to_read:
    classes = list(to_read.read().strip())

num_classes = len(classes) +1 
print(num_classes)
model = build_model(num_classes,256, 1)
model.load_weights(restore, by_name=True, skip_mismatch=True)

model.save("crnn_tf")
