# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:54:09 2024

@author: lich5
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import make_moons
import tensorflow as tf
from tensorflow.keras import layers, callbacks
import types
import math


# from svm_tensorflow import *

import sys
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class ShowProgress(callbacks.Callback):
    def __init__(self, epochs, step_show=1, metric="accuracy"):
        super(ShowProgress, self).__init__()
        self.epochs = epochs
        self.step_show = step_show
        self.metric = metric

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(range(self.epochs))

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.step_show == 0:

            self.pbar.set_description(f"""Epoch : {epoch + 1} / {self.epochs}, 
            Train {self.metric} : {round(logs[self.metric], 4)}, 
            Valid {self.metric} : {round(logs['val_' + self.metric], 4)}""")

            self.pbar.update(self.step_show)

            
class BestModelWeights(callbacks.Callback):
    def __init__(self, metric="val_accuracy", metric_type="max"):
        super(BestModelWeights, self).__init__()
        self.metric = metric
        self.metric_type = metric_type
        if self.metric_type not in ["min", "max"]:
                raise NameError('metric_type must be min or max')

    def on_train_begin(self, logs=None):
        if self.metric_type == "min":
            self.best_metric = math.inf
        else:
            self.best_metric = -math.inf
        self.best_epoch = 0
        self.model_best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        if self.metric_type == "min":
            if self.best_metric >= logs[self.metric]:
                self.model_best_weights = self.model.get_weights()
                self.best_metric = logs[self.metric]
                self.best_epoch = epoch
        else:
            if self.best_metric <= logs[self.metric]:
                self.model_best_weights = self.model.get_weights()
                self.best_metric = logs[self.metric]
                self.best_epoch = epoch

    def on_train_end(self, logs=None):
        self.model.set_weights(self.model_best_weights)
        print(f"\nBest weights is set, Best Epoch was : {self.best_epoch+1}\n")

#classes
class LinearSVC(layers.Layer):
    def __init__(self, num_classes=2, **kwargs):
        super(LinearSVC, self).__init__(**kwargs)
        self.num_classes = num_classes
    
        self.reg_loss = lambda weight : 0.5 * tf.reduce_sum(tf.square(weight))

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.num_classes),
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.num_classes,), initializer=tf.constant_initializer(value=0.1),
            trainable=True
        )

    def call(self, inputs):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.reg_loss(self.w)
        self.add_loss(loss)
        
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(LinearSVC, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config




class SVMTrainer(tf.keras.Model):
    def __init__(
        self,
        num_class,
        C=1.0,
        bone=None,
        name="SVMTrainer",
        **kwargs
    ):
        super(SVMTrainer, self).__init__(name=name, **kwargs)
    
        self.num_class = num_class

        if bone is None:
            self.bone = lambda x: tf.identity(x)
        else:
            self.bone = bone

        self.linear_svc = LinearSVC(self.num_class)
        self.C = C
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    
    def svc_loss(self, y_true, y_pred, sample_weight, reg_loss):
        
        loss = tf.keras.losses.squared_hinge(y_true ,y_pred)
        if sample_weight is not None:
            loss = sample_weight * loss
        
        return reg_loss + self.C * loss
    
    
    def compile(self, **kwargs):
        super(SVMTrainer, self).compile(**kwargs)
        self.compiled_loss = None
    
    
    def call(self, x, training=False):
        x = self.bone(x)
        x = self.linear_svc(x)
        return x

    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.svc_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                reg_loss=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        if self.num_class == 2:
            y = y[..., 1]
            y_pred = tf.sigmoid(y_pred[..., 1])

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    
    def test_step(self, data):
        # Unpack the data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss = self.svc_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                reg_loss=self.losses,
        )
        # Update the metrics.
        if self.num_class == 2:
            y = y[..., 1]
            y_pred = tf.sigmoid(y_pred[..., 1])
        
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch or at the start of `evaluate()`.
        return [self.loss_tracker] + self.compiled_metrics.metrics


    def save(self, model_path=None, input_shape=None):
        input_shape = [1] + input_shape 
        dumy_input = np.random.rand(*input_shape)


        dumy_body_output = self.bone(dumy_input)
        dumy_head_output = self.linear_svc(dumy_body_output)


        head_part = layers.Dense(units=dumy_head_output.shape[-1], activation="sigmoid")
        _ = head_part(dumy_body_output)
        head_part.set_weights(self.linear_svc.get_weights())


        if isinstance(self.bone, types.FunctionType):
            body_part = layers.Lambda(lambda x: self.bone(x))
        else:
            body_part = self.bone


        input_shape.pop(0)
        inputs = layers.Input(shape=input_shape)
        x = body_part(inputs)
        x = head_part(x)


        model = tf.keras.models.Model(inputs, x)
        model.save(model_path)



if __name__ == '__main__':
    # Define Data
    data = make_moons(3000, noise=0.05)
    x, y = data
    y = tf.one_hot(y, depth=2, on_value=1, off_value=0).numpy()
    
    x, y = shuffle(x, y)
    
    n_train = int(0.8 * len(x))
    train_x, train_y = x[:n_train], y[:n_train]
    valid_x, valid_y = x[n_train:], y[n_train:]
    
    # Define metrics
    METRICS = [
          tf.keras.metrics.TruePositives(name='tp'),
          tf.keras.metrics.FalsePositives(name='fp'),
          tf.keras.metrics.TrueNegatives(name='tn'),
          tf.keras.metrics.FalseNegatives(name='fn'),
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.AUC(name='auc'),
          tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    
    # Define Bone, if you want linear svm, you can pass None to SVMTrainer as bone
    Bone = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(128, activation="relu"),
    ])
    
    
    svm_model = SVMTrainer(num_class=2, bone=Bone)
    svm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                      metrics=METRICS)
    
    # Callbacks
    epochs = 200
    show_progress = ShowProgress(epochs)
    best_weight = BestModelWeights()
    
    # Train
    history = svm_model.fit(train_x, train_y,
                            epochs=epochs, validation_data=(valid_x, valid_y),
                            callbacks=[show_progress, best_weight],
                            verbose=0 # When you want to use ShowProgress callback, you should set verbose to zero
                                )
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Helper function for plot metrics
    def plot_metrics(history):
        plt.figure(figsize=(12, 10))
        metrics = ['loss', 'prc', 'accuracy', 'fp', 'precision', "tp", "recall", "tn", "auc", "fn"]
    
        for n, metric in enumerate(metrics):
    
            name = metric.replace("_"," ").capitalize()
            plt.subplot(5, 2, n+1)
    
            plt.plot(history.epoch,
                     history.history[metric],
                     color=colors[0],
                     label='Train')
    
            plt.plot(history.epoch,
                     history.history['val_'+ metric],
                     color=colors[1],
                     #linestyle="--",
                     label='Val')
    
            plt.xlabel('Epoch')
            plt.ylabel(name)
    
            plt.legend()
    
    # plot_metrics(history)
    
    
    plt.figure(figsize=(15, 10))
    Min = x.min(axis=0)
    Max = x.max(axis=0)
    
    a = np.linspace(Min[0], Max[0], 200)
    b = np.linspace(Min[1], Max[1], 200)
    xa, xb = np.meshgrid(a, b)
    
    X = np.stack([xa, xb], axis=-1)
    X = np.reshape(X, [-1, 2])
    
    bound = svm_model.predict(X)
    bound = np.argmax(bound, axis=-1)
    
    class1 = X[bound == 0]
    class2 = X[bound == 1]
    
    plt.scatter(class1[:,0], class1[:,1])
    plt.scatter(class2[:,0], class2[:,1])
    
    plt.scatter(x[:,0], x[:,1])
