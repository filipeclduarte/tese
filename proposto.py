# proposto recursivo

import pandas as pd
import numpy as np
import time
from datetime import datetime as dt
import sys
from scipy.interpolate import CubicSpline

sys.path.append('./')
from models.mylstm import get_best_trained_model as get_best_trained_model_lstm
from models.mymlp import get_best_trained_model as get_best_trained_model_mlp
from models.mynbeats import get_best_trained_model as get_best_trained_model_nbeats
# from pmdarima.arima import auto_arima
from utils import divisao_dados_temporais, normalize_serie, desnormalize, split_sequence, reshape_X, forecast_rnn, forecast_mlp, forecast_nbeats
# from telegram_bot import ExpBot
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import tensorflow as tf
from tensorflow.keras import backend as K
import gc
import argparse


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

np.random.seed(1)
tf.random.set_seed(1)

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


def main(country, gender, init_year):
    # bot_message = ExpBot()
    # bot_message.send_message(f'[INFO] HyS-MF - Country: {country} | Gender: {gender} | Initial Year: {init_year}')
    print(f'[INFO] Country: {country} | Gender: {gender} | Initial Year: {init_year}')
    predictions_arima = pd.read_csv(f'./results/univariate/predictions_arima_{country}_{gender}.csv', sep=',')
    predictions_arima_training = pd.read_csv(f'./results/predictions_training_arima_{country}_{gender}.csv', sep=',')
    residuals_series = pd.read_csv(f'./results/residuals_{country}_{gender}.csv', sep=',')
    lnmx_series = pd.read_csv(f'./data/{country}_{gender}.csv', sep = ',', index_col = 0)
    ages = ['0', '1', '2', '5', '10', '12', '15', '18','20', '22', '25',
            '28', '30', '40', '50', '60', '70', '80', '90', '100']

    predictions_arima_df = predictions_arima[ages]
    residuals_series_df = residuals_series[ages]

    # train and test
    # train = lnmx_series.loc[init_year:2009, ages]
    test = lnmx_series.loc[2010:2019, ages]
    curve_train = lnmx_series.loc[init_year:2009, '0':'100']
    curve_series = lnmx_series.loc[init_year:2019, '0':'100']
    lnmx_series = lnmx_series.loc[init_year:2019, ages]
    
    # ajustar interpolação portugal
    if country == 'portugal' and gender == 'male':
        temp = curve_series.loc[2015].copy()
        temp.index = temp.index.astype(int)
        curve_series.loc[2015] = temp.interpolate(method='spline', order=3).values
        
        temp = curve_series.loc[2017].copy()
        temp.index = temp.index.astype(int)
        curve_series.loc[2017] = temp.interpolate(method='spline', order=3).values

    if country == 'portugal' and gender == 'female':
        curve_series_mx = curve_series.transpose()
        m_ = curve_series_mx[curve_series_mx[2011] == -np.inf].loc[:, [2010, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]].mean(1).values[0]
        curve_series_mx[2011].replace(-np.inf, m_, inplace=True)
        
        m_ = curve_series_mx[curve_series_mx[2019] == -np.inf].loc[:, 2010:2018].mean(1).values[0]
        curve_series_mx[2019].replace(-np.inf, m_, inplace=True)
        curve_series = curve_series_mx.transpose()

    
    # alocating arrays
    predictions_ages_lstm = np.zeros((10,20))
    predictions_ages_val_lstm = dict.fromkeys(ages)
    rmse_hys_lstm = np.zeros((1,2))
    mape_hys_lstm = np.zeros((1,2))
    lstm_hp = dict.fromkeys(ages)
    
    predictions_ages_mlp = np.zeros((10,20))
    predictions_ages_val_mlp = dict.fromkeys(ages)
    rmse_hys_mlp = np.zeros((1,2))
    mape_hys_mlp = np.zeros((1,2))
    mlp_hp = dict.fromkeys(ages)

    predictions_ages_nbeats = np.zeros((10,20))
    predictions_ages_val_nbeats = dict.fromkeys(ages)
    rmse_hys_nbeats = np.zeros((1,2))
    mape_hys_nbeats = np.zeros((1,2))
    nbeats_hp = dict.fromkeys(ages)


    # n_lags = 2
    for age, i in zip(ages, range(0,len(ages)+1)):
        n_lags = 2

        print(f'AGE: {age}')

        # train_i = train[age]
        residual_serie_i = residuals_series_df[age]
        arima_predict = predictions_arima_df[age]
        arima_predict_training_i = predictions_arima_training[age]
        test_i = test[age]
                
        # Training with hybrid but analyze after using the test result
        arima_error_norm = normalize_serie(residual_serie_i.values)
    
        # series to supervised n_lags and 1 output 
        X_error, y_error = split_sequence(arima_error_norm, n_steps_in=n_lags, n_steps_out=1)
        X_train, y_train, X_val, y_val  = divisao_dados_temporais(X_error, y_error, perc_treino = 0.8)
        
        # training and optimizing hyperparameters with bayes
        # calculate elapsed time - olhometro: aprox 20-40s por configuração (trial)
        # reshape to 3d
        X_train_, X_val_ = reshape_X(X_train, X_val)
        model_lstm, lstm_hp_ = get_best_trained_model_lstm(X_train_.shape[1], X_train_, y_train, X_val_, y_val)
        lstm_hp[age] = lstm_hp_.values['units']
        model_mlp, mlp_hp_ = get_best_trained_model_mlp(X_train.shape[1], X_train, y_train, X_val, y_val)
        mlp_hp[age] = mlp_hp_.values['units']
        model_nbeats, nbeats_hp_ = get_best_trained_model_nbeats(X_train_.shape[1], X_train_, y_train, X_val_, y_val)
        nbeats_hp[age] = nbeats_hp_.values['units']

        # validation
        pred_lstm_val = forecast_rnn(model_lstm, X_train_[-1,:].ravel(), 2, y_val.shape[0]).ravel()
        pred_mlp_val = forecast_mlp(model_mlp, X_train_[-1,:].ravel(), 2, y_val.shape[0]).ravel()
        pred_nbeats_val = forecast_nbeats(model_nbeats, X_train_[-1,:].ravel(), 2, y_val.shape[0]).ravel()

        pred_lstm_val = desnormalize(pred_lstm_val, residual_serie_i.values)
        pred_mlp_val = desnormalize(pred_mlp_val, residual_serie_i.values)
        pred_nbeats_val = desnormalize(pred_nbeats_val, residual_serie_i.values)

        hys_lstm_val = arima_predict_training_i.values[-pred_lstm_val.shape[0]:] + pred_lstm_val.values
        hys_mlp_val = arima_predict_training_i.values[-pred_lstm_val.shape[0]:] + pred_mlp_val.values
        hys_nbeats_val = arima_predict_training_i.values[-pred_lstm_val.shape[0]:] + pred_nbeats_val.values
        predictions_ages_val_lstm[age] = hys_lstm_val
        predictions_ages_val_mlp[age] = hys_mlp_val
        predictions_ages_val_nbeats[age] = hys_nbeats_val
        
        # prediction
        X_test_pred = arima_error_norm[-n_lags:].ravel()
        predictions_lstm = forecast_rnn(model_lstm, X_test_pred, n_lags, 10).ravel()
        predictions_mlp = forecast_mlp(model_mlp, X_test_pred, n_lags, 10).ravel()
        predictions_nbeats = forecast_nbeats(model_nbeats, X_test_pred, n_lags, 10).ravel()
    

        # free mem
        del model_lstm, model_mlp, model_nbeats
        K.clear_session()
        gc.collect()

        # reverse normalization
        predictions_lstm = desnormalize(predictions_lstm, residual_serie_i.values)
        predictions_mlp = desnormalize(predictions_mlp, residual_serie_i.values)
        predictions_nbeats = desnormalize(predictions_nbeats, residual_serie_i.values)

        # hybrid system linear + nonlinear
        predictions_lstm = arima_predict.values.ravel() + predictions_lstm.ravel() 
        predictions_mlp = arima_predict.values.ravel() + predictions_mlp.ravel() 
        predictions_nbeats = arima_predict.values.ravel() + predictions_nbeats.ravel() 

        predictions_ages_lstm[:, i] = predictions_lstm
        predictions_ages_mlp[:, i] = predictions_mlp
        predictions_ages_nbeats[:, i] = predictions_nbeats 
 

    # Saving
    predictions_ages_lstm = pd.DataFrame(predictions_ages_lstm, columns = ages)
    predictions_ages_val_lstm = pd.DataFrame(predictions_ages_val_lstm)
    predictions_ages_mlp = pd.DataFrame(predictions_ages_mlp, columns = ages)
    predictions_ages_val_mlp = pd.DataFrame(predictions_ages_val_mlp)
    predictions_ages_nbeats = pd.DataFrame(predictions_ages_nbeats, columns=ages)
    predictions_ages_val_nbeats = pd.DataFrame(predictions_ages_val_nbeats)
    
    lstm_hp = pd.DataFrame(lstm_hp.values()).transpose()
    lstm_hp.columns = ages
    mlp_hp = pd.DataFrame(mlp_hp.values()).transpose()
    mlp_hp.columns = ages
    nbeats_hp = pd.DataFrame(nbeats_hp.values()).transpose()
    nbeats_hp.columns = ages

    ## create the curve for val and test
    predictions_curve_val_lstm = interpolacao_curva(predictions_ages_val_lstm, ages, 
                                                    jan_previsao=predictions_ages_val_lstm.shape[0])
    predictions_curve_val_lstm.columns = curve_train.index[-predictions_curve_val_lstm.shape[1]:]
    rmse_hys_lstm[0, 0] = MSE(curve_train.transpose().iloc[:, -predictions_ages_val_lstm.shape[0]:].values.ravel(),
                            predictions_curve_val_lstm.values.ravel(), squared=False)
    mape_hys_lstm[0,0] = MAPE(curve_train.transpose().iloc[:, -predictions_ages_val_lstm.shape[0]:].values.ravel(),
                            predictions_curve_val_lstm.values.ravel()) * 100

    predictions_curve_lstm = interpolacao_curva(predictions_ages_lstm, ages)
    predictions_curve_lstm.columns = test.index
    rmse_hys_lstm[0, 1] = MSE(curve_series.transpose().iloc[:, -10:].values.ravel(),
                            predictions_curve_lstm.values.ravel(), squared=False)
    mape_hys_lstm[0, 1] = MAPE(curve_series.transpose().iloc[:, -10:].values.ravel(),
                            predictions_curve_lstm.values.ravel()) * 100

    predictions_curve_val_mlp = interpolacao_curva(predictions_ages_val_mlp, ages, 
                                                   jan_previsao=predictions_ages_val_mlp.shape[0])
    predictions_curve_val_mlp.columns = curve_train.index[-predictions_curve_val_mlp.shape[1]:]
    rmse_hys_mlp[0, 0] = MSE(curve_train.transpose().iloc[:, -predictions_ages_val_mlp.shape[0]:].values.ravel(),
                            predictions_curve_val_mlp.values.ravel(), squared=False)
    mape_hys_mlp[0,0] = MAPE(curve_train.transpose().iloc[:, -predictions_ages_val_mlp.shape[0]:].values.ravel(),
                            predictions_curve_val_mlp.values.ravel()) * 100

    predictions_curve_mlp = interpolacao_curva(predictions_ages_mlp, ages)
    predictions_curve_mlp.columns = test.index
    rmse_hys_mlp[0, 1] = MSE(curve_series.transpose().iloc[:, -10:].values.ravel(),
                            predictions_curve_mlp.values.ravel(), squared=False)
    mape_hys_mlp[0, 1] = MAPE(curve_series.transpose().iloc[:, -10:].values.ravel(),
                            predictions_curve_mlp.values.ravel()) * 100
    

    predictions_curve_val_nbeats = interpolacao_curva(predictions_ages_val_nbeats, ages, 
                                                      jan_previsao=predictions_ages_val_nbeats.shape[0])
    predictions_curve_val_nbeats.columns = curve_train.index[-predictions_curve_val_nbeats.shape[1]:]
    rmse_hys_nbeats[0, 0] = MSE(curve_train.transpose().iloc[:, -predictions_ages_val_nbeats.shape[0]:].values.ravel(),
                            predictions_curve_val_nbeats.values.ravel(), squared=False)
    mape_hys_nbeats[0,0] = MAPE(curve_train.transpose().iloc[:, -predictions_ages_val_nbeats.shape[0]:].values.ravel(),
                            predictions_curve_val_nbeats.values.ravel()) * 100
    
    predictions_curve_nbeats = interpolacao_curva(predictions_ages_nbeats, ages)
    predictions_curve_nbeats.columns = test.index
    rmse_hys_nbeats[0, 1] = MSE(curve_series.transpose().iloc[:, -10:].values.ravel(),
                            predictions_curve_nbeats.values.ravel(), squared=False)
    mape_hys_nbeats[0, 1] = MAPE(curve_series.transpose().iloc[:, -10:].values.ravel(),
                            predictions_curve_nbeats.values.ravel()) * 100


    rmse_hys_lstm = pd.DataFrame(rmse_hys_lstm, columns = ['val', 'test'], index=['lstm'])
    mape_hys_lstm = pd.DataFrame(mape_hys_lstm, columns = ['val', 'test'], index=['lstm'])
    rmse_hys_mlp = pd.DataFrame(rmse_hys_mlp, columns = ['val', 'test'], index=['mlp'])
    mape_hys_mlp = pd.DataFrame(mape_hys_mlp, columns = ['val', 'test'], index=['mlp'])
    rmse_hys_nbeats = pd.DataFrame(rmse_hys_nbeats, columns = ['val', 'test'], index=['nbeats'])
    mape_hys_nbeats = pd.DataFrame(mape_hys_nbeats, columns = ['val', 'test'], index=['nbeats'])


    proposed_error = pd.DataFrame({'rmse_val':[rmse_hys_lstm.iloc[0,0], rmse_hys_mlp.iloc[0,0], rmse_hys_nbeats.iloc[0,0]],
                          'mape_val': [mape_hys_lstm.iloc[0,0], mape_hys_mlp.iloc[0,0], mape_hys_nbeats.iloc[0,0]],
                          'rmse_test': [rmse_hys_lstm.iloc[0,1], rmse_hys_mlp.iloc[0,1], rmse_hys_nbeats.iloc[0,1]],
                          'mape_test': [mape_hys_lstm.iloc[0,1], mape_hys_mlp.iloc[0,1], mape_hys_nbeats.iloc[0,1]],
                          }, index=['lstm', 'mlp', 'nbeats'])
    
    model_selected = proposed_error['mape_val'].idxmin() # ou rmse
    proposed_curve = predictions_curve_lstm.copy()
    if model_selected == 'mlp':
        proposed_curve = predictions_curve_mlp.copy()
    elif model_selected == 'nbeats':
        proposed_curve = predictions_curve_nbeats.copy()

    # Exportando os resultados para .csv
    proposed_curve.to_csv(f'./results/univariate/predictions_proposed_val_{country}_{gender}.csv', index=None, header=True, encoding = 'utf-8')
    predictions_curve_lstm.to_csv(f'./results/univariate/predictions_arima_lstm_{country}_{gender}.csv', index=None, header=True, encoding = 'utf-8')
    predictions_curve_mlp.to_csv(f'./results/univariate/predictions_arima_mlp_{country}_{gender}.csv', index=None, header=True, encoding = 'utf-8')
    predictions_curve_nbeats.to_csv(f'./results/univariate/predictions_arima_nbeats_{country}_{gender}.csv', index=None, header=True, encoding = 'utf-8')
    proposed_error.to_csv(f'./results/proposed_error_{country}_{gender}.csv', header=True, encoding = 'utf-8')
    proposed_error.rank().to_csv(f'./results/proposed_error_rank_{country}_{gender}.csv', header=True, encoding = 'utf-8')
    # residuals_series_nonlinear.to_json(f'./results/residuals_series_linear_nonlinear_{country}_{gender}.json')

    return model_selected

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
    model_selected = main(args.country, args.gender, args.init_year)
    toc = time.time()
    exec_time = dt.strftime(dt.utcfromtimestamp(toc-tic), '%H:%M:%S')
    with open(f"./proposed_ages_execution_time.txt", "a") as f:
        f.write(f'[INFO] PROPOSED | MODEL: {model_selected} | {args.country} | {args.gender} | Execution time: {exec_time}' + '\n')

