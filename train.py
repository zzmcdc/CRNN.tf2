import argparse
import time
import os
import tensorflow as tf
from tensorflow import keras
from dataset import DatasetBuilder
from model import build_model
from losses import CTCLoss
from metrics import WordAccuracy
from tensorflow.keras import backend as K
import numpy as np

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


class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler

        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.

        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.

    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.

    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.

    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.

    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


dataset_builder = DatasetBuilder(
    args.charset, args.img_width, args.img_channels, args.ignore_case)
train_ds, train_size = dataset_builder.build(
    args.train_ann_paths, True, args.batch_size)

print('Num of training samples: {}'.format(train_size))
print("num of label", dataset_builder.num_classes)
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

model = build_model(dataset_builder.num_classes,
                    args.img_width, channels=args.img_channels)
model.compile(optimizer=keras.optimizers.SGD(args.learning_rate, momentum=0.9,  clipnorm=1),
              loss=CTCLoss(), metrics=[WordAccuracy()])

if args.restore:
    model.load_weights(args.restore, by_name=True, skip_mismatch=True)

epoch_batch = 975000/args.batch_size

warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=args.learning_rate,
                                        total_steps=args.epochs * epoch_batch,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=epoch_batch,
                                        hold_base_rate_steps=10 * epoch_batch)


callbacks = [warm_up_lr,
             tf.keras.callbacks.TerminateOnNaN(),
             keras.callbacks.ModelCheckpoint(saved_model_path),
             keras.callbacks.TensorBoard(log_dir='logs/{}'.format(localtime), profile_batch=0)]
model.fit(train_ds, epochs=args.epochs, callbacks=callbacks,
          validation_data=val_ds)
