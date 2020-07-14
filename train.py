import argparse
import time
import os

from tensorflow import keras

from dataset import DatasetBuilder
from model import build_model
from losses import CTCLoss
from metrics import WordAccuracy

parser = argparse.ArgumentParser()
parser.add_argument('-ta', '--train_ann_paths', type=str,
                    required=True,
                    help='The path of training data annnotation file.')
parser.add_argument('-va', '--val_ann_paths', type=str,
                    help='The path of val data annotation file.')
parser.add_argument('-w', '--img_width', type=int, default=100,
                    help='Image width, this parameter will affect the output '
                         'shape of the model, default is 100, so this model '
                         'can only predict up to 24 characters.')
parser.add_argument('-b', '--batch_size', type=int, default=256,
                    help='Batch size.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('-e', '--epochs', type=int, default=30,
                    help='Num of epochs to train.')
parser.add_argument('--img_channels', type=int, default=1,
                    help='0: Use the number of channels in the image, '
                         '1: Grayscale image, 3: RGB image')
parser.add_argument('--ignore_case', action='store_true',
                    help='Whether ignore case.(default false)')
parser.add_argument('--restore', type=str,
                    help='The model for restore, even if the number of '
                         'characters is different')
parser.add_argument('--charset', type=str,
                    default='label.txt', help='the charset file')
args = parser.parse_args()

localtime = time.asctime()

with open(args.charset) as to_read:
    classes = list(to_read.read().strip())
num_classes = len(classes)

dataset_builder = DatasetBuilder(
    args.charset, args.img_width, args.img_channels, args.ignore_case)
train_ds, train_size = dataset_builder.build(
    args.train_ann_paths, True, args.batch_size)

print('Num of training samples: {}'.format(train_size))
saved_model_prefix = '{epoch:03d}_{word_accuracy:.4f}'
if args.val_ann_paths:
    val_ds, val_size = dataset_builder.build(
        args.val_ann_paths, False, args.batch_size)
    print('Num of val samples: {}'.format(val_size))
    saved_model_prefix = saved_model_prefix + '_{val_word_accuracy:.4f}'
else:
    val_ds = None
saved_model_path = ('saved_models/{}/'.format(localtime) +
                    saved_model_prefix + '.h5')
os.makedirs('saved_models/{}'.format(localtime))
print('Training start at {}'.format(localtime))

model = build_model(num_classes, channels=args.img_channels)

#model = multi_gpu_model(model, gpus=4)


model.compile(optimizer=keras.optimizers.Adam(args.learning_rate, clipvalue=1),
              loss=CTCLoss(), metrics=[WordAccuracy()])

if args.restore:
    model.load_weights(args.restore, by_name=True, skip_mismatch=True)

callbacks = [keras.callbacks.ModelCheckpoint(saved_model_path),
             keras.callbacks.TensorBoard(log_dir='logs/{}'.format(localtime),
                                         profile_batch=0)]
model.fit(train_ds, epochs=args.epochs, callbacks=callbacks,
          validation_data=val_ds)
