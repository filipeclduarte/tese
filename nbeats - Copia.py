import pandas as pd
import numpy as np
import time
from datetime import datetime as dt
import sys
sys.path.append('./')
from models.mynbeats import get_best_trained_model
from utils import divisao_dados_temporais, normalize_serie, desnormalize, split_sequence, reshape_X, forecast_nbeats
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import tensorflow as tf
from tensorflow.keras import backend as K
import gc
import argparse
from utils import interpolacao_curva


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

np.random.seed(1)
tf.random.set_seed(1)

def main(country, gender, init_year):
    print(f'[INFO] Country: {country} | Gender: {gender} | Initial Year: {init_year}')
    lnmx_series = pd.read_csv(f'./data/{country}_{gender}.csv', sep = ',', index_col = 0)
    ages = ['0', '1', '2', '5', '10', '12', '15', '18','20', '22', '25',
            '28', '30', '40', '50', '60', '70', '80', '90', '100']


    # train and test
    train = lnmx_series.loc[init_year:2009, ages]
    test = lnmx_series.loc[2010:2019, ages]
    lnmx_series = lnmx_series.loc[init_year:2019, ages]
    # alocating arrays
    predictions_ages = np.zeros((10,20))
    rmse_nbeats = np.zeros((1,20))
    mape_nbeats = np.zeros((1, 20))
    n_lags = 2

    for age, i in zip(ages, range(0,len(ages)+1)):
        # lnmx = lnmx_series[age]
        test_i = test[age]
        train_i = train[age]
        
        # Training with hybrid but analyze after using the test result
        serie_norm = normalize_serie(train_i.values)
    
        # series to supervised n_lags and 1 output 
        X, y = split_sequence(serie_norm, n_steps_in=n_lags, n_steps_out=1)
        X_train, y_train, X_val, y_val  = divisao_dados_temporais(X, y, perc_treino = 0.8)
        X_train_, X_val_ = reshape_X(X_train, X_val)
        model, _ = get_best_trained_model(X_train_.shape[1], X_train_, y_train, X_val_, y_val)

        # prediction
        X_test_pred = serie_norm[-n_lags:].ravel()
        predictions = forecast_nbeats(model, X_test_pred, n_lags, 10).ravel()

        # free mem
        del model
        K.clear_session()
        gc.collect()

        # reverse normalization
        predictions = desnormalize(predictions, train_i.values).ravel()

        predictions_ages[:, i] = predictions
        # RMSE
        rmse_nbeats[0,i] = MSE(test_i.values, predictions, squared=False)
        mape_nbeats[0,i] = MAPE(test_i.values, predictions) * 100 

        print(f"[INFO] NBEATS | AGE: {age} | RMSE: {rmse_nbeats[0,i]}")
        print(f"[INFO] NBEATS | AGE: {age} | MAPE: {mape_nbeats[0,i]}")

    # Saving
    predictions_ages = interpolacao_curva(predictions_ages)
  
    # Exportando os resultados para .csv
    predictions_ages.to_csv(f'./results/univariate/predictions_nbeats_{country}_{gender}.csv', index=None, header=True, encoding = 'utf-8')
    

if __name__ == '__main__':

    # # creating the arg's parser
    parser = argparse.ArgumentParser(description='Select the country and the gender of the population')
    parser.add_argument('country', metavar='P', type=str, help='country: australia, belgica, \
                         canada, espanha, eua, french, japao, portugal, suecia, suica or uk')
    parser.add_argument('gender', metavar='G', type=str, help='gender of population: female, male or total')
    parser.add_argument('init_year', metavar='IY', type=int, help='initial year: 1950')

    # parse the args
    args = parser.parse_args()
    tic = time.time()
    # execution of the main function
    main(args.country, args.gender, args.init_year)
    toc = time.time()
    exec_time = dt.strftime(dt.utcfromtimestamp(toc-tic), '%H:%M:%S')
    with open(f"./nbeats_ages_execution_time.txt", "a") as f:
        f.write(f'[INFO] NBEATS | {args.country} | {args.gender} | Execution time: {exec_time}' + '\n')
    