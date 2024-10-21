import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
import sys
sys.path.append('./')
from utils import model_executions


class MyMLP(kt.HyperModel):
    def __init__(self, seq_len, outputs):
        self.seq_len = seq_len
        self.outputs = outputs
    
    def build(self, hp):
        units = hp.Int('units', min_value=2, max_value=100)
        activation_fun = hp.Choice('activation_fun', values=['tanh','relu'])
        # dropout = hp.Float('dropout', 0.0, 0.8, step=0.1, default=0.5)
        model = keras.Sequential([
            layers.Dense(units, activation=activation_fun),
            # layers.Dropout(dropout),
            layers.Dense(self.outputs)
        ])
        lr = hp.Float('learning_rate', 1e-4, 1e-1, sampling='log', default=1e-3)

        # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss='mean_squared_error', # learning_rate = 0.001
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error',
        metrics=['mean_squared_error'])
        return model


def build_tuner(hypermodel):
    objective = kt.Objective(
        name='val_loss',
        direction='min'
    )

    tuner = kt.BayesianOptimization(
    hypermodel,
    objective=objective,
    max_trials=10, # testando com 10 por enquanto
    executions_per_trial=3,
    directory='bayesian_tuning_MLP',
    overwrite=True,
    seed=0
    )
    return tuner


def get_best_hp(seq_len, X_train, Y_train, X_val, Y_val):
    hypermodel = MyMLP(seq_len, Y_train.shape[1])
    tuner = build_tuner(hypermodel)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss',patience=5),
    ]
    tuner.search(X_train, Y_train, 
                        validation_data=(X_val, Y_val),
                        epochs=200, callbacks=callbacks, verbose=2,
                        batch_size=8,  
                        # use_multiprocessing=True, workers=4
    )
    best_hp = tuner.get_best_hyperparameters()[0]
    return best_hp

def get_best_epoch(hp, seq_len, X_train, Y_train, X_val, Y_val):
    model = MyMLP(seq_len, Y_train.shape[1]).build(hp)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss',patience=10, mode='min'),
    ]
    history = model.fit(X_train, Y_train, epochs=500, 
                        validation_data=(X_val, Y_val),
                        callbacks=callbacks,
                        batch_size=8,
                        )
    val_loss_per_epoch = history.history['val_loss']
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    return best_epoch

def get_best_trained_model(seq_len, X_train, Y_train, X_val, Y_val, num_exec=10):
    best_hp = get_best_hp(seq_len, X_train, Y_train, X_val, Y_val)
    best_epoch = get_best_epoch(best_hp, seq_len, X_train, Y_train, X_val, Y_val)
    model = MyMLP(seq_len, Y_train.shape[1]).build(best_hp)
    # best_epoch = int(best_epoch * 1.2)
    print(f'BEST EPOCH: {best_epoch}')
    # 10 executions and get the best result
    model = model_executions(model, X_train, Y_train, X_val, Y_val, best_epoch, batch_size=8, num_exec=num_exec)

    return model, best_hp
