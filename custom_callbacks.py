#how to stop at some epochs if some accuracy reached ?
#use callbaks
import tensorflow as tf

class custom_accuracy_callback(tf.keras.callbacks.Callback):
    def __init__(self, accuracy):
        self._accuracy = accuracy
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > self._accuracy:
            self.model.stop_training = True


class custom_loss_callback(tf.keras.callbacks.Callback):
    def __init__(self, loss):
        self._loss = loss
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < self._loss:
            self.model.stop_training = True


