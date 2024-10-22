import pandas as pd
import numpy as np
import sys
sys.path.append('./')
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import argparse
import time
from utils import interpolacao_curva

np.random.seed(1)

# TODO: salvar series residuais e erro de validacao
def main(country, gender, init_year):
    print(f'[INFO] Country: {country} | Gender: {gender} | Initial Year: {init_year}')
    lnmx_series = pd.read_csv(f'./data/{country}_{gender}.csv', sep = ',', index_col = 0)
    print(lnmx_series)
    ages = ['0', '1', '2', '5', '10', '12', '15', '18','20', '22', '25',
            '28', '30', '40', '50', '60', '70', '80', '90', '100']
    
    # train and test
    train = lnmx_series.loc[init_year:2009, ages]
    test = lnmx_series.loc[2010:2019, ages]
    lnmx_series = lnmx_series.loc[init_year:2019, ages]
    # ages
    ages = train.columns
    # alocating arrays
    predictions_ages = np.zeros((10,20)) 
    rmse_arima = np.zeros((1, 20))
    mape_arima = np.zeros((1, 20))


    for age, i in zip(ages, range(0,len(ages)+1)):
        lnmx = lnmx_series[age]
        train_i = train[age]
        test_i = test[age]

        # ARIMA
        arima = auto_arima(train_i, start_p=1, start_q=1,
                            max_p=10, max_q=10, m=12,
                            start_P=0, seasonal=False,
                            d=1, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
        
        # train
        arima.fit(train_i)
       
        # arima forecast
        arima_predict = arima.predict(n_periods=10, return_conf_int=False)
        # arima_predict = pd.Series(arima_predict, index = lnmx.loc[2010:2019].index)
        # linear
        predictions = arima_predict.values.ravel() 
        print(f'PREDICTIONS: {predictions}')

        predictions_ages[:, i] = predictions 
        # RMSE
        rmse_arima[0,i] = MSE(test_i.values, predictions, squared=False)
        mape_arima[0,i] = MAPE(test_i.values, predictions) * 100

        print(f"[INFO] ARIMA | AGE: {age} | RMSE: {rmse_arima[0,i]}")
        print(f"[INFO] ARIMA | AGE: {age} | MAPE: {mape_arima[0,i]}")


    # Saving
    # predictions_ages = pd.DataFrame(predictions_ages, columns = ages)
    predictions_ages = interpolacao_curva(predictions_ages, ages)
    predictions_ages.columns = test.index

    # rmse_arima = pd.DataFrame(rmse_arima, columns = ages)
    # mape_arima = pd.DataFrame(mape_arima, columns = ages)
    
    # Exportando os resultados para .csv
    predictions_ages.to_csv(f'./results/predictions_arima_{country}_{gender}.csv', index=None, header=True, encoding = 'utf-8')
    # rmse_arima.to_csv(f'./results/rmse_arima_{country}_{gender}.csv', index=None, header=True, encoding = 'utf-8')
    # mape_arima.to_csv(f'./results/mape_arima_{country}_{gender}.csv', index=None, header=True, encoding = 'utf-8')
    

if __name__ == '__main__':
    from datetime import datetime as dt
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
    exec_time = dt.strftime(dt.utcfromtimestamp(toc-tic), '%H:%M:%S')
    with open(f"./arima_execution_time.txt", "a") as f:
        f.write(f'[INFO] ARIMA | {args.country} | {args.gender} | Execution time: {exec_time}' + '\n')
