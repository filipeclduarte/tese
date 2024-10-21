import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow import keras
from nbeats_keras.model import NBeatsNet as NBeatsKeras
from scipy.interpolate import CubicSpline


def interpolacao_curva(dados,  x_idades, qnt_idades = 101, jan_previsao = 10):
    '''
    Faz a interpolação das curvas de mortalidade previstas
    Parâmetros:
        dados: previsões dos modelos para as idades
        x_idades: idades
        qnt_idades: quantidade total de idades
        jan_previsao: janela de previsão
    Retorna:
        curva: curva de mortalidade conténdo todas as idades (colunas: janela de previsão)
    '''
    curva = np.zeros((qnt_idades, jan_previsao)) 
    x = x_idades
    xs = np.arange(0, qnt_idades, 1)
    for i in range(0,jan_previsao):
        y = dados.iloc[i,:]
        cs = CubicSpline(x, y)
        curva[:,i] = cs(xs)  
    curva = pd.DataFrame(curva, index = xs)
    curva.columns = [i for i in range(1,jan_previsao+1)] 
    return curva



def divisao_dados_temporais(X,y, perc_treino, perc_val = 0):
    tam_treino = int(perc_treino * len(y))
    
    if perc_val > 0:        
        tam_val = int(len(y)*perc_val)
              
        X_treino = X[0:tam_treino,:]
        y_treino = y[0:tam_treino,:]
        
        print("Training size:", 0, tam_treino)
        
        X_val = X[tam_treino:tam_treino+tam_val,:]
        y_val = y[tam_treino:tam_treino+tam_val,:]
        
        print("Validation size:",tam_treino,tam_treino+tam_val)
        
        X_teste = X[(tam_treino+tam_val):,:]
        y_teste = y[(tam_treino+tam_val):,:]
        
        print("Test size:", tam_treino+tam_val, len(y))
        
        return X_treino, y_treino, X_teste, y_teste, X_val, y_val
        
    else:
        
        X_treino = X[0:tam_treino,:]
        y_treino = y[0:tam_treino,:]

        X_teste = X[tam_treino:,:]
        y_teste = y[tam_treino:,:]

        return X_treino, y_treino, X_teste, y_teste 


def normalize_serie(serie):
    minimo = min(serie)
    maximo = max(serie)
    y = (serie - minimo) / (maximo - minimo)
    return y

def desnormalize(serie_atual, serie_real):
    minimo = min(serie_real)
    maximo = max(serie_real)
    
    serie = (serie_atual * (maximo - minimo)) + minimo
    
    return pd.Series(serie)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# reshape 2d to 3d
def reshape_X(X_train, X_val):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    return X_train, X_val


# forecast recursive for mlp
def forecast_mlp(model, input_x, n_input, n_steps):
	yhat_sequence = list()
	input_data = [x for x in input_x]
	for j in range(n_steps):
		# prepare the input data
		X = np.array(input_data[-n_input:]).reshape(1, n_input)
		# make a one-step forecast
		yhat = model.predict(X)[0,0]
		# add to the result
		yhat_sequence.append(yhat)
		# add the prediction to the input
		input_data.append(yhat)
	return np.array(yhat_sequence)

def forecast_rnn(model, input_x, n_input, n_steps):
    yhat_sequence = list()
    input_data = [x for x in input_x]
    for j in range(n_steps):
        # prepare the input data
        X = np.array(input_data[-n_input:]).reshape(1, n_input, 1)
        # make a one-step forecast
        yhat = model.predict(X)[0,0]
        # add to the result
        yhat_sequence.append(yhat)
        # add the prediction to the input
        input_data.append(yhat)
    return np.array(yhat_sequence)

def forecast_cnnrnn(model, input_x, n_input, n_steps):
    yhat_sequence = list()
    input_data = [x for x in input_x]
    for j in range(n_steps):
        # prepare the input data
        X = np.array(input_data[-n_input:]).reshape(1, int(n_input/2), int(n_input/2), 1)
        # make a one-step forecast
        yhat = model.predict(X)[0,0]
        # add to the result
        yhat_sequence.append(yhat)
        # add the prediction to the input
        input_data.append(yhat)
    return np.array(yhat_sequence)


def forecast_nbeats(model, input_x, n_input, n_steps):
    yhat_sequence = list()
    input_data = [x for x in input_x]
    for j in range(n_steps):
        # prepare the input data
        X = np.array(input_data[-n_input:]).reshape(1, n_input, 1)
        # make a one-step forecast
        yhat = model.predict(X)[0,0,0]
        # add to the result
        yhat_sequence.append(yhat)
        # add the prediction to the input
        input_data.append(yhat)
    return np.array(yhat_sequence)



def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer': #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            var.assign(initializer(var.shape, var.dtype))
            #use the initializer


def model_executions(model, X_train, y_train, X_val, y_val, epochs, batch_size, num_exec):
    
    weights_list = []
    val_mse_list = []
    
    for i in range(num_exec):
        print(f'EXECUTION: {i+1}')
        # training
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        # storing 
        val_mse_ = model.evaluate(X_val, y_val)[1]
        val_mse_list.append(val_mse_)
        weights_ = model.get_weights()
        weights_list.append(weights_)
        
        # reset weights
        reset_weights(model)

    # get the best model
    best_mse = min(val_mse_list)
    best_mse_idx = val_mse_list.index(best_mse)
    print(f'BEST EXECUTION: {best_mse_idx+1}')
    best_weights = weights_list[best_mse_idx]
    # set the best result
    model.set_weights(best_weights)
    
    return model


def model_executions_nbeats(my_model, seq_len, best_hp, X_train, y_train, X_val, y_val, epochs, batch_size, num_exec):
    
    val_mse_list = []
    models_list = []
    
    for i in range(num_exec):
        print(f'EXECUTION: {i+1}')
        # clear_session()
        model = my_model(seq_len, y_train.shape[1]).build(best_hp)
        # training
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # storing 
        val_mse_ = model.evaluate(X_val, y_val)[1]
        val_mse_list.append(val_mse_)
        
        models_list.append(model)

    # get the best model
    best_mse = min(val_mse_list)
    best_mse_idx = val_mse_list.index(best_mse)
    print(f'BEST EXECUTION: {best_mse_idx+1}')
    model = models_list[best_mse_idx]

    return model

def model_executions_nbeats2(my_model, best_hp, X_train, y_train, X_val, y_val, epochs, num_exec):
    
    val_mse_list = []
    models_list = []
    seq_len = best_hp.values['seq_len']
    for i in range(num_exec):
        print(f'EXECUTION: {i+1}')
        # clear_session()
        model =  my_model(
            backcast_length=best_hp.values['seq_len'], forecast_length=y_train.shape[1],
            stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
            nb_blocks_per_stack=best_hp.values['nb_blocks'], thetas_dim=(4, 4), share_weights_in_stack=True,
            hidden_layer_units=best_hp.values['units']
        )
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',patience=10, mode='min')]
        optimizer = keras.optimizer=keras.optimizers.Adam(learning_rate=best_hp.values['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
        # training
        model.fit(X_train[:, -seq_len:], y_train, 
                  epochs=epochs, 
                  batch_size=best_hp.values['batch_size'],
                  callbacks=callbacks,
                  validation_data=(X_val[:, -seq_len:], y_val))
        # storing 
        val_mse_ = model.evaluate(X_val[:, -seq_len:], y_val)[1]
        val_mse_list.append(val_mse_)
        
        models_list.append(model)

    # get the best model
    best_mse = min(val_mse_list)
    best_mse_idx = val_mse_list.index(best_mse)
    print(f'BEST EXECUTION: {best_mse_idx+1}')
    model = models_list[best_mse_idx]

    return model, seq_len
