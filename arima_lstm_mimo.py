# ARIMA-lstm MIMO

import pandas as pd
import numpy as np
import time
from datetime import datetime as dt
import sys
sys.path.append('./')
from models.mylstm import get_best_trained_model
# from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import bds
from scipy.stats import wilcoxon
from utils import divisao_dados_temporais, normalize_serie, desnormalize, split_sequence, reshape_X
from telegram_bot import ExpBot
from iafft import surrogates
from ordpy import permutation_entropy
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

# TODO: salvar series residuais e erro de validacao
def main(country, gender, init_year):
    # bot_message = ExpBot()
    # bot_message.send_message(f'[INFO] ARIMA_lstm MIMO - Country: {country} | Gender: {gender} | Initial Year: {init_year}')
    print(f'[INFO] Country: {country} | Gender: {gender} | Initial Year: {init_year}')
    predictions_arima = pd.read_csv(f'./results/univariate/predictions_arima_2_{country}_{gender}.csv', sep=',')
    residuals_series = pd.read_csv(f'./results/residuals_2{country}_{gender}.csv', sep=',')
    lnmx_series = pd.read_csv(f'./data/{country}_{gender}.csv', sep = ',', index_col = 0)
    ages = ['0', '1', '2', '5', '10', '12', '15', '18','20', '22', '25',
            '28', '30', '40', '50', '60', '70', '80', '90', '100']

    predictions_arima_df = predictions_arima[ages]
    residuals_series_df = residuals_series[ages]

    # train and test
    # train = lnmx_series.loc[init_year:2009, ages]
    test = lnmx_series.loc[2010:2019, ages]
    lnmx_series = lnmx_series.loc[init_year:2019, ages]
    # alocating arrays
    predictions_ages = np.zeros((10,20))
    rmse_hys = np.zeros((1,20))
    mape_hys = np.zeros((1, 20))
    # save residuals series that contains nonlinear patterns
    # residuals_series_nonlinear = dict.fromkeys(ages)
    # n_lags = 2

    for age, i in zip(ages, range(0,len(ages)+1)):
        n_lags = 10
        # lnmx = lnmx_series[age]
        residual_serie_i = residuals_series_df[age]
        arima_predict = predictions_arima_df[age]
        test_i = test[age]

        # # BDS test
        # _, arima_error_bds_test = bds(residual_serie_i.values, 3)
        # # surrogates test
        # ## conditional evaluation: if residuals contain nonlinear patterns
        # ### generate surrogates
        # surrogates_qtd = int((2/0.05) - 1)
        # residuals_linear_surrogates = surrogates(residual_serie_i.values, surrogates_qtd)
        # ### calculate entropy permutation
        # arima_error_pe = permutation_entropy(residual_serie_i)
        # surrogates_pe = np.apply_along_axis(permutation_entropy, 1, residuals_linear_surrogates)
        # arima_error_rank_test = wilcoxon(surrogates_pe, arima_error_pe)
        
        # ## decision for continue modeling the residuals 
        # #TODO: if at least one rejection of linearity, continue the process
        # message_info_teste = f'[INFO] Country: {country} | gender: {gender} | age: {age} | p-value BDS test: {arima_error_bds_test[-1]} | p-value Rank test PE surrogates: {arima_error_rank_test.pvalue}'
        # print(message_info_teste)
        # # bot_message.send_message(message_info_teste)
        # if((arima_error_bds_test[-1] or arima_error_rank_test.pvalue) < 0.05):
        #     # continue the nonlinear modeling
        #     residuals_series_nonlinear[age] = 'nonlinear'
        # else:
        #     # nonlinear equals to 0.0
        #     residuals_series_nonlinear[age] = 'linear'
        #     # predictions = pd.Series(np.zeros(10))
        
        # Training with hybrid but analyze after using the test result
        arima_error_norm = normalize_serie(residual_serie_i.values)
    
        # series to supervised n_lags and 1 output 
        X_error, y_error = split_sequence(arima_error_norm, n_steps_in=n_lags, n_steps_out=10)
        X_train, y_train, X_val, y_val  = divisao_dados_temporais(X_error, y_error, perc_treino = 0.8)
        
        # reshape to 3d
        X_train_, X_val_ = reshape_X(X_train, X_val)
        # training and optimizing hyperparameters with bayes
        # model = get_best_trained_model(X_train_.shape[1], X_train_, y_train, X_val_, y_val)
        model, n_lags = get_best_trained_model(X_train_, y_train, X_val_, y_val)
        # prediction
        X_test_pred = arima_error_norm[-n_lags:].reshape(1, n_lags, 1)
        predictions = model.predict(X_test_pred).ravel()

        # free mem
        del model
        K.clear_session()
        gc.collect()

        # reverse normalization
        predictions = desnormalize(predictions, residual_serie_i.values)
        # predictions.index = test_i.index

        # hybrid system linear + nonlinear
        predictions = arima_predict.values.ravel() + predictions.ravel() 

        predictions_ages[:, i] = predictions 
        # RMSE
        rmse_hys[0,i] = MSE(test_i.values, predictions, squared=False)
        mape_hys[0,i] = MAPE(test_i.values, predictions) * 100 

        print(f"[INFO] ARIMA-LSTM MIMO | AGE: {age} | RMSE: {rmse_hys[0,i]}")
        print(f"[INFO] ARIMA-LSTM MIMO | AGE: {age} | MAPE: {mape_hys[0,i]}")

    # Saving
    predictions_ages = interpolacao_curva(predictions_ages, ages)
    # Exportando os resultados para .csv
    predictions_ages.to_csv(f'./results/univariate/predictions_arima_lstm_mimo_{country}_{gender}.csv', index=None, header=True, encoding = 'utf-8')
    

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
    # bot_m = ExpBot()
    # exec_time = dt.strftime(dt.utcfromtimestamp(toc-tic), '%H:%M:%S')
    # try:
    #     bot_m.send_message(f'[INFO] ARIMA-LSTM MIMO | {args.country} | {args.gender} | Execution time: {exec_time}')
    # except:
    #     print('An exception has occured')
    with open(f"./arima_lstm_mimo_ages_execution_time.txt", "a") as f:
        f.write(f'[INFO] ARIMA-LSTM MIMO | {args.country} | {args.gender} | Execution time: {exec_time}' + '\n')
