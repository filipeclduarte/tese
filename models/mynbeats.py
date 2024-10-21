import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
# from kerasbeats import NBeats
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

import sys
sys.path.append('./')
from nbeats_keras.model import NBeatsNet as NBeatsKeras
from utils import model_executions_nbeats


class MyNBeats(kt.HyperModel):
    def __init__(self, seq_len, outputs):
        self.seq_len = seq_len
        self.outputs = outputs
    
    def build(self, hp):
        units = hp.Int('units', min_value=2, max_value=100)
        # nb_blocks = hp.Int('nb_blocks', min_value=2, max_value=4)
        # share_weights_in_satck = hp.Choice('share_weights_in_stack', [True, False])
        theta_dim_b = hp.Int('theta_dim_b', min_value=1, max_value=4)
        # theta_dim_f = hp.Int('theta_dim_f', min_value=1, max_value=8)
        # dropout = hp.Float('dropout', 0.0, 0.8, step=0.1, default=0.5)
        model = NBeatsKeras(
            backcast_length=self.seq_len, forecast_length=self.outputs,
            stack_types=(NBeatsKeras.GENERIC_BLOCK,), #, NBeatsKeras.TREND_BLOCK),
            nb_blocks_per_stack=4, # 4 é o valor conforme recomendação se for genérico
            # nb_blocks_per_stack=nb_blocks, 
            # thetas_dim=(theta_dim_b, theta_dim_f), 
            thetas_dim=(theta_dim_b,), #generic usa apenas 1 dim
            share_weights_in_stack= False, ### MODELO GENÉRICO NÃO USA CONFORME ESTUDO #share_weights_in_satck, #true
            hidden_layer_units=units,
            # , 
            # dropout=None
        )

        lr = hp.Float('learning_rate', 1e-4, 1e-1, sampling='log', default=1e-3)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error',
                    metrics=["mean_squared_error"])
        # model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.005),
        return model


def build_tuner(hypermodel):
    objective = kt.Objective(name='val_loss', direction='min')
    tuner = kt.BayesianOptimization(hypermodel,
                                    objective=objective,
                                    max_trials=10, # testando com 10 por enquanto
                                    executions_per_trial=3, # 5 by default
                                    directory='bayesian_tuning_lstm',
                                    overwrite=True,
                                    seed=0)
    return tuner


def get_best_hp(seq_len, X_train, Y_train, X_val, Y_val):
    hypermodel = MyNBeats(seq_len, Y_train.shape[1])
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
    model = MyNBeats(seq_len, Y_train.shape[1]).build(hp)
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
    # model = MyNBeats(seq_len, Y_train.shape[1]).build(best_hp)
    best_epoch = int(best_epoch * 1.2)
    print(f'BEST EPOCH: {best_epoch}')
    # X_train_val = np.vstack((X_train, X_val))
    # Y_train_val = np.vstack((Y_train, Y_val))
    # model.fit(X_train_val, Y_train_val, epochs=best_epoch, batch_size=8)
    # 10 executions and get the best result
    model = model_executions_nbeats(MyNBeats, seq_len, best_hp, X_train, Y_train, X_val, Y_val, best_epoch, 
                                    batch_size=8, num_exec=num_exec)
    return model, best_hp
