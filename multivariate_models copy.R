### script to execute the hys-mf with multivariate mortality forecasting models
library(rjson)
library(forecast)
library(demography)
library(StMoMo)
source('hmd_mx.R')
source('read_predictions_r.R')

data_list = list('french'= c('FRATNP', 1950),
                 'portugal' = c('PRT', 1950),
                 'australia' = c('AUS', 1921),
                 'eua' = c('USA', 1950),
                 'japan' = c('JPN', 1950))
# download dados
## funcoes
read_hmd_mx_disk <- function(country_name, country)
{
  return(hmd_mx_disk(country_name, country))
}

filter_years <- function(data_country, init_year, last_year=2009){
  return(extract.years(data_country, years = init_year:last_year))
}
load_data_disk <- function(data_list, test=FALSE){
  
  data_countries <- lapply(seq_along(data_list), function(y, i) {read_hmd_mx_disk(names(y[i]), y[[i]][1])},
                                                             y=data_list)
  names(data_countries) = names(data_list)
  
  if(test) 
  {
    init_year = 2010
    last_year = 2019
    data_train <- lapply(seq_along(data_countries), function(y, n, i) {filter_years(y[[i]], init_year, last_year)}, 
                         y=data_countries,
                         n=names(data_list))
    
  } else {
    data_train <- lapply(seq_along(data_countries), function(y, n, i) {filter_years(y[[i]], data_list[[i]][2])}, 
                         y=data_countries,
                         n=names(data_list))
  }
  
  names(data_train) = names(data_list)
  return(data_train)
}

# data
data_countries <- load_data_disk(data_list)
# test data
data_countries_test <- load_data_disk(data_list, TRUE)

# treinamento de modelos 
train <- function(model, data, gender="total")
{
  data_train <- StMoMoData(data, series=gender)
  ages.fit <- 0:100
  wxt <- genWeightMat(ages = ages.fit, years = data_train$years, 
                      clip=3) # cohort with at least 3 years
  mod <- fit(model, data=data_train, ages.fit=ages.fit, wxt=wxt)
  return(mod)
}

## LC
LC <- lc(link="log")
start_time <- Sys.time()
lc_models_total <- lapply(data_countries, function(m, x) {train(m, x)}, m=LC)
lc_models_male <- lapply(data_countries, function(m, x) {train(m, x, "male")}, m=LC)
lc_models_female <- lapply(data_countries, function(m, x) {train(m, x, "female")}, m=LC)
end_time <- Sys.time()
execution_time <- end_time - start_time
print(execution_time)
  
## PLAT - Combination of LC and CBD with cohort effects
f2 <- function(x, ages) mean(ages) - x
constPlat <- function(ax, bx, kt, b0x, gc, wxt, ages) {
  nYears <- dim(wxt)[2]
  x <- ages
  t <- 1:nYears
  c <- (1 - tail(ages, 1)):(nYears - ages[1])
  xbar <- mean(x)
  phiReg <- lm(gc ~ 1 + c + I(c ^ 2), na.action = na.omit)
  phi <- coef(phiReg)
  gc <- gc - phi[1] - phi[2] * c - phi[3] * c ^ 2
  kt[2, ] <- kt[2, ] + 2 * phi[3] * t
  kt[1, ] <- kt[1, ] + phi[2] * t + phi[3] * (t ^ 2 - 2 * xbar * t)
  ax <- ax + phi[1] - phi[2] * x + phi[3] * x ^ 2
  ci <- rowMeans(kt, na.rm = TRUE)
  ax <- ax + ci[1] + ci[2] * (xbar - x)
  kt[1, ] <- kt[1, ] - ci[1]
  kt[2, ] <- kt[2, ] - ci[2]
  list(ax = ax, bx = bx, kt = kt, b0x = b0x, gc = gc)}

start_time <- Sys.time()
PLAT <- StMoMo(link = "log", staticAgeFun = TRUE,
              periodAgeFun = c("1", f2), cohortAgeFun = "1", constFun = constPlat)
plat_models_total <- lapply(data_countries, function(m, x){train(m, x)}, m=PLAT)  
plat_models_male <- lapply(data_countries, function(m, x){train(m, x, "male")}, m=PLAT)  
plat_models_female <- lapply(data_countries, function(m, x){train(m, x, "female")}, m=PLAT)  
end_time <- Sys.time()
execution_time <- end_time - start_time
print(execution_time)

# previsoes multivariados classicos
lc_forecast_total <- lapply(lc_models_total, function(x, h) {forecast(x, h=10)})
lc_forecast_male <- lapply(lc_models_male, function(x, h) {forecast(x, h=10)})
lc_forecast_female <- lapply(lc_models_female, function(x, h) {forecast(x, h=10)})

plat_forecast_total <- lapply(plat_models_total, function(x, h) {forecast(x, h=10)})
plat_forecast_male <- lapply(plat_models_male, function(x, h) {forecast(x, h=10)})
plat_forecast_female <- lapply(plat_models_female, function(x, h) {forecast(x, h=10)})


# salvando previsoes multivariados classicos
write_predictions <- function(predictions, model_name, country, gender){
  write.csv(predictions, paste0('./results/predictions_', model_name, '_', country, '_', gender, '.csv'))
}
lapply(seq_along(lc_forecast_total), function(x, n, c, i){write_predictions(t(log(x[[i]]$rates)), n, c[i], 'total')}, 
                x=lc_forecast_total, n='lc',c=names(lc_forecast_total))
lapply(seq_along(lc_forecast_male), function(x, n, c, i){write_predictions(t(log(x[[i]]$rates)), n, c[i], 'male')}, 
       x=lc_forecast_male, n='lc',c=names(lc_forecast_male))
lapply(seq_along(lc_forecast_female), function(x, n, c, i){write_predictions(t(log(x[[i]]$rates)), n, c[i], 'female')}, 
       x=lc_forecast_female, n='lc',c=names(lc_forecast_female))

## plat
lapply(seq_along(plat_forecast_total), function(x, n, c, i){write_predictions(t(log(x[[i]]$rates)), n, c[i], 'total')}, 
       x=plat_forecast_total, n='plat',c=names(plat_forecast_total))
lapply(seq_along(plat_forecast_male), function(x, n, c, i){write_predictions(t(log(x[[i]]$rates)), n, c[i], 'male')}, 
       x=plat_forecast_male, n='plat',c=names(plat_forecast_male))
lapply(seq_along(plat_forecast_female), function(x, n, c, i){write_predictions(t(log(x[[i]]$rates)), n, c[i], 'female')}, 
       x=plat_forecast_female, n='plat',c=names(plat_forecast_female))

#TODO: salvando fatores 
write_factors <- function(factor, model_name, factor_name, country, gender){
       write.csv(factor, paste0('./results/', model_name, '_', factor_name, '_', country ,'_', gender, '.csv'))
}

lapply(seq_along(lc_models_total), function(x, n, fn, c, i){write_factors(t(x[[i]]$kt), n, fn, c[i], 'total')},
       x=lc_models_total, n='lc', fn='kt', c=names(lc_models_total))
lapply(seq_along(lc_models_male), function(x, n, fn, c, i){write_factors(t(x[[i]]$kt), n, fn, c[i], 'male')},
       x=lc_models_male, n='lc', fn='kt', c=names(lc_models_male))
lapply(seq_along(lc_models_female), function(x, n, fn, c, i){write_factors(t(x[[i]]$kt), n, fn, c[i], 'female')},
       x=lc_models_female, n='lc', fn='kt', c=names(lc_models_female))


# plat
lapply(seq_along(plat_models_total), function(x, n, fn, c, i){write_factors(x[[i]]$kt[1, ], n, fn, c[i], 'total')},
       x=plat_models_total, n='plat', fn='kt1', c=names(plat_models_total))
lapply(seq_along(plat_models_male), function(x, n, fn, c, i){write_factors(x[[i]]$kt[1, ], n, fn, c[i], 'male')},
       x=plat_models_male, n='plat', fn='kt1',c=names(plat_models_male))
lapply(seq_along(plat_models_female), function(x, n, fn, c, i){write_factors(x[[i]]$kt[1, ], n, fn, c[i], 'female')},
       x=plat_models_female, n='plat', fn='kt1',c=names(plat_models_female))
lapply(seq_along(plat_models_total), function(x, n, fn, c, i){write_factors(x[[i]]$kt[2, ], n, fn, c[i], 'total')},
       x=plat_models_total, n='plat', fn='kt2',c=names(plat_models_total))
lapply(seq_along(plat_models_male), function(x, n, fn, c, i){write_factors(x[[i]]$kt[2, ], n, fn, c[i], 'male')},
       x=plat_models_male, n='plat', fn='kt2',c=names(plat_models_male))
lapply(seq_along(plat_models_female), function(x, n, fn, c, i){write_factors(x[[i]]$kt[2, ], n, fn, c[i], 'female')},
       x=plat_models_female, n='plat', fn='kt2', c=names(plat_models_female))
lapply(seq_along(plat_models_total), function(x, n, fn, c, i){write_factors(x[[i]]$gc, n, fn, c[i], 'total')},
       x=plat_models_total, n='plat', fn='gc',c=names(plat_models_total))
lapply(seq_along(plat_models_male), function(x, n, fn, c, i){write_factors(x[[i]]$gc, n, fn, c[i], 'male')},
       x=plat_models_male, n='plat', fn='gc',c=names(plat_models_male))
lapply(seq_along(plat_models_female), function(x, n, fn, c, i){write_factors(x[[i]]$gc, n, fn, c[i], 'female')},
       x=plat_models_female, n='plat', fn='gc', c=names(plat_models_female))


### auto arima nos fatores e salvando residuos e previsoes
train_arima_f <- function(i)
{
  cat(i, '\n')
  # lc
  lc_kt_arima_model_total <- auto.arima(t(lc_models_total[[i]]$kt))
  lc_kt_arima_residuals_total <- residuals(lc_kt_arima_model_total)
  lc_kt_forecast_arima_model_total <- forecast(lc_kt_arima_model_total, h=10)$mean
  
  lc_kt_arima_model_male <- auto.arima(t(lc_models_male[[i]]$kt))
  lc_kt_arima_residuals_male <- residuals(lc_kt_arima_model_male)
  lc_kt_forecast_arima_model_male <- forecast(lc_kt_arima_model_male, h=10)$mean

  lc_kt_arima_model_female <- auto.arima(t(lc_models_female[[i]]$kt))
  lc_kt_arima_residuals_female <- residuals(lc_kt_arima_model_female)
  lc_kt_forecast_arima_model_female <- forecast(lc_kt_arima_model_female, h=10)$mean
  
  # plat
  plat_kt1_arima_model_total <- auto.arima(plat_models_total[[i]]$kt[1, ])
  plat_kt1_arima_residuals_total <- residuals(plat_kt1_arima_model_total)
  plat_kt1_forecast_arima_model_total <- forecast(plat_kt1_arima_model_total, h=10)$mean
  plat_kt2_arima_model_total <- auto.arima(plat_models_total[[i]]$kt[2, ])
  plat_kt2_arima_residuals_total <- residuals(plat_kt2_arima_model_total)
  plat_kt2_forecast_arima_model_total <- forecast(plat_kt2_arima_model_total, h=10)$mean
  plat_gc_arima_model_total <- auto.arima(ts(plat_models_total[[i]]$gc))
  plat_gc_arima_residuals_total <- residuals(plat_gc_arima_model_total)
  plat_gc_forecast_arima_model_total <- forecast(plat_gc_arima_model_total, h=10)$mean # pq temos 0 missing
  
  plat_kt1_arima_model_male <- auto.arima(plat_models_male[[i]]$kt[1, ])
  plat_kt1_arima_residuals_male <- residuals(plat_kt1_arima_model_male)
  plat_kt1_forecast_arima_model_male <- forecast(plat_kt1_arima_model_male, h=10)$mean
  plat_kt2_arima_model_male <- auto.arima(plat_models_male[[i]]$kt[2, ])
  plat_kt2_arima_residuals_male <- residuals(plat_kt2_arima_model_male)
  plat_kt2_forecast_arima_model_male <- forecast(plat_kt2_arima_model_male, h=10)$mean
  plat_gc_arima_model_male <- auto.arima(ts(plat_models_male[[i]]$gc))
  plat_gc_arima_residuals_male <- residuals(plat_gc_arima_model_male)
  plat_gc_forecast_arima_model_male <- forecast(plat_gc_arima_model_male, h=10)$mean # pq temos 0 missing
  
  plat_kt1_arima_model_female <- auto.arima(plat_models_female[[i]]$kt[1, ])
  plat_kt1_arima_residuals_female <- residuals(plat_kt1_arima_model_female)
  plat_kt1_forecast_arima_model_female <- forecast(plat_kt1_arima_model_female, h=10)$mean
  plat_kt2_arima_model_female <- auto.arima(plat_models_female[[i]]$kt[2, ])
  plat_kt2_arima_residuals_female <- residuals(plat_kt2_arima_model_female)
  plat_kt2_forecast_arima_model_female <- forecast(plat_kt2_arima_model_female, h=10)$mean
  plat_gc_arima_model_female <- auto.arima(ts(plat_models_female[[i]]$gc))
  plat_gc_arima_residuals_female <- residuals(plat_gc_arima_model_female)
  plat_gc_forecast_arima_model_female <- forecast(plat_gc_arima_model_female, h=10)$mean # pq temos 0 missing
  
  # # write
  ## lc
  write.csv(lc_kt_arima_residuals_total, paste0('./results/residuals_lc_kt_', i, '_total.csv'))
  write.csv(lc_kt_forecast_arima_model_total, paste0('./results/predictions_lc_kt_', i, '_total.csv'))
  
  write.csv(lc_kt_arima_residuals_male, paste0('./results/residuals_lc_kt_', i, '_male.csv'))
  write.csv(lc_kt_forecast_arima_model_male, paste0('./results/predictions_lc_kt_', i, '_male.csv'))
  
  write.csv(lc_kt_arima_residuals_female, paste0('./results/residuals_lc_kt_', i, '_female.csv'))
  write.csv(lc_kt_forecast_arima_model_female, paste0('./results/predictions_lc_kt_', i, '_female.csv'))
   
  ## plat
  write.csv(plat_kt1_arima_residuals_total, paste0('./results/residuals_plat_kt1_', i, '_total.csv'))
  write.csv(plat_kt1_forecast_arima_model_total, paste0('./results/predictions_plat_kt1_', i, '_total.csv'))
  write.csv(plat_kt2_arima_residuals_total, paste0('./results/residuals_plat_kt2_', i, '_total.csv'))
  write.csv(plat_kt2_forecast_arima_model_total, paste0('./results/predictions_plat_kt2_', i, '_total.csv'))
  write.csv(plat_gc_arima_residuals_total, paste0('./results/residuals_plat_gc_', i, '_total.csv'))
  write.csv(plat_gc_forecast_arima_model_total, paste0('./results/predictions_plat_gc_', i, '_total.csv'))
  
  write.csv(plat_kt1_arima_residuals_male, paste0('./results/residuals_plat_kt1_', i, '_male.csv'))
  write.csv(plat_kt1_forecast_arima_model_male, paste0('./results/predictions_plat_kt1_', i, '_male.csv'))
  write.csv(plat_kt2_arima_residuals_male, paste0('./results/residuals_plat_kt2_', i, '_male.csv'))
  write.csv(plat_kt2_forecast_arima_model_male, paste0('./results/predictions_plat_kt2_', i, '_male.csv'))
  write.csv(plat_gc_arima_residuals_male, paste0('./results/residuals_plat_gc_', i, '_male.csv'))
  write.csv(plat_gc_forecast_arima_model_male, paste0('./results/predictions_plat_gc_', i, '_male.csv'))
  
  write.csv(plat_kt1_arima_residuals_female, paste0('./results/residuals_plat_kt1_', i, '_female.csv'))
  write.csv(plat_kt1_forecast_arima_model_female, paste0('./results/predictions_plat_kt1_', i, '_female.csv'))
  write.csv(plat_kt2_arima_residuals_female, paste0('./results/residuals_plat_kt2_', i, '_female.csv'))
  write.csv(plat_kt2_forecast_arima_model_female, paste0('./results/predictions_plat_kt2_', i, '_female.csv'))
  write.csv(plat_gc_arima_residuals_female, paste0('./results/residuals_plat_gc_', i, '_female.csv'))
  write.csv(plat_gc_forecast_arima_model_female, paste0('./results/predictions_plat_gc_', i, '_female.csv'))
  
  return(
    list(
      'lc_kt_arima_total' = lc_kt_arima_model_total,
      'lc_kt_forecast_total'= lc_kt_forecast_arima_model_total,
      'lc_kt_residuals_total' = lc_kt_arima_residuals_total,
      'lc_kt_arima_male' = lc_kt_arima_model_male,
      'lc_kt_forecast_male'= lc_kt_forecast_arima_model_male,
      'lc_kt_residuals_male' = lc_kt_arima_residuals_male,
      'lc_kt_arima_female' = lc_kt_arima_model_female,
      'lc_kt_forecast_female'= lc_kt_forecast_arima_model_female,
      'lc_kt_residuals_female' = lc_kt_arima_residuals_female,
      'plat_kt1_arima_total'=plat_kt1_arima_model_total,
      'plat_kt1_forecast_total'=plat_kt1_forecast_arima_model_total,
      'plat_kt1_residuals_total'=plat_kt1_arima_residuals_total,
      'plat_kt2_arima_total'=plat_kt2_arima_model_total,
      'plat_kt2_forecast_total'=plat_kt2_forecast_arima_model_total,
      'plat_kt2_residuals_total'=plat_kt2_arima_residuals_total,
      'plat_gc_arima_total'=plat_gc_arima_model_total,
      'plat_gc_forecast_total'=plat_gc_forecast_arima_model_total,
      'plat_gc_residuals_total'=plat_gc_arima_residuals_total,
      'plat_kt1_arima_male'=plat_kt1_arima_model_male,
      'plat_kt1_forecast_male'=plat_kt1_forecast_arima_model_male,
      'plat_kt1_residuals_male'=plat_kt1_arima_residuals_male,
      'plat_kt2_arima_male'=plat_kt2_arima_model_male,
      'plat_kt2_forecast_male'=plat_kt2_forecast_arima_model_male,
      'plat_kt2_residuals_male'=plat_kt2_arima_residuals_male,
      'plat_gc_arima_male'=plat_gc_arima_model_male,
      'plat_gc_forecast_male'=plat_gc_forecast_arima_model_male,
      'plat_gc_residuals_male'=plat_gc_arima_residuals_male,
      'plat_kt1_arima_female'=plat_kt1_arima_model_female,
      'plat_kt1_forecast_female'=plat_kt1_forecast_arima_model_female,
      'plat_kt1_residuals_female'=plat_kt1_arima_residuals_female,
      'plat_kt2_arima_female'=plat_kt2_arima_model_female,
      'plat_kt2_forecast_female'=plat_kt2_forecast_arima_model_female,
      'plat_kt2_residuals_female'=plat_kt2_arima_residuals_female,
      'plat_gc_arima_female'=plat_gc_arima_model_female,
      'plat_gc_forecast_female'=plat_gc_forecast_arima_model_female,
      'plat_gc_residuals_female'=plat_gc_arima_residuals_female
  ))
}
arima_models <- lapply(names(lc_models_total), train_arima_f)
names(arima_models) <- names(data_list)

# from 2010 - 100 = 1900
# to 2019 - 0 = 2019
### salvando previsao multivariado com auto arima
predict_write_multivariate_arima_models <- function(i){
  cat('[INFO] country: ', i, '\n')
  years_forecast <- 2010:2019
  mx_lc_total <- predict(lc_models_total[[i]], 
                 years=years_forecast, 
                 kt=arima_models[[i]]$lc_kt_forecast_total, type='rates')
  
  mx_lc_male <- predict(lc_models_male[[i]], 
                   years=years_forecast, 
                   kt=arima_models[[i]]$lc_kt_forecast_male, type='rates')
  
  mx_lc_female <- predict(lc_models_female[[i]], 
                   years=years_forecast, 
                   kt=arima_models[[i]]$lc_kt_forecast_female, type='rates')
    
  mx_plat_total <- predict(object=plat_models_total[[i]],
                     years=years_forecast,
                     kt=rbind(arima_models[[i]]$plat_kt1_forecast_total,
                              arima_models[[i]]$plat_kt2_forecast_total),
                     gc=c(na.omit(tail(plat_models_total[[i]]$gc, 101+2)), 
                          arima_models[[i]]$plat_gc_forecast_total),
                     type='rates')
  
  mx_plat_male <- predict(object=plat_models_male[[i]],
                     years=years_forecast,
                     kt=rbind(arima_models[[i]]$plat_kt1_forecast_male,
                              arima_models[[i]]$plat_kt2_forecast_male),
                     gc=c(na.omit(tail(plat_models_male[[i]]$gc, 101+2)), 
                          arima_models[[i]]$plat_gc_forecast_male),
                     type='rates')
  
  mx_plat_female <- predict(object=plat_models_female[[i]],
                     years=years_forecast,
                     kt=rbind(arima_models[[i]]$plat_kt1_forecast_female,
                              arima_models[[i]]$plat_kt2_forecast_female),
                     gc=c(na.omit(tail(plat_models_female[[i]]$gc, 101+2)), 
                          arima_models[[i]]$plat_gc_forecast_female),
                     type='rates')
  
  # write csv
  write.csv(t(mx_lc_total), paste0('./results/predictions_lc_arima_', i, '_total.csv'), row.names=F)
  write.csv(t(mx_lc_male), paste0('./results/predictions_lc_arima_', i, '_male.csv'), row.names=F)
  write.csv(t(mx_lc_female), paste0('./results/predictions_lc_arima_', i, '_female.csv'), row.names=F)
  
  write.csv(t(mx_plat_total), paste0('./results/predictions_plat_arima_', i, '_total.csv'),row.names=F)
  write.csv(t(mx_plat_male), paste0('./results/predictions_plat_arima_', i, '_male.csv'),row.names=F)
  write.csv(t(mx_plat_female), paste0('./results/predictions_plat_arima_', i, '_female.csv'),row.names=F)
  
  return(list(
    'lc_arima_total'=mx_lc_total,
    'lc_arima_male'=mx_lc_male,
    'lc_arima_female'=mx_lc_female,
    'plat_arima_total'=mx_plat_total,
    'plat_arima_male'=mx_plat_male,
    'plat_arima_female'=mx_plat_female
    )
  )
}
predictions_multivariate_arima <- lapply(names(arima_models), predict_write_multivariate_arima_models)
names(predictions_multivariate_arima) <- names(data_list)


# Executando sistema hibrido codigo em python (voltar para esse script) 
# execute_hys_mf_multivariate <- function(names_models){
# for (i in names_models){
#   system(paste('python scripts/hys_mf_multivariate.py', i, 'total'))
# }
# }
# execute_hys_mf_multivariate(names(lc_models))

# leitura da previsao sistema hibrido dos fatores
fatores_ <- c('lc_kt', 'plat_kt1', 'plat_kt2', 'plat_gc')

predictions_df <- load_predictions()

load_predictions_res_mult <- function()
{
  predictions_df <- load_predictions()

  predictions_df_mult <- function(predictions_df)
  {
    predictions_list <- list()
    for(i in 1:length(predictions_df))
    {
      if(grepl('predictions_residuals_multivariate', names(predictions_df)[i]) | 
         grepl('predictions_residuals_arima', names(predictions_df)[i]) |
         # grepl('mlp_predictions', names(predictions_df)[i]) |
         startsWith(names(predictions_df)[i],'mlp_') |
         startsWith(names(predictions_df)[i],'lstm_') |
         startsWith(names(predictions_df)[i],'nbeats_'))
         # grepl('mlp_mimo_predictions', names(predictions_df)[i]) |
         # grepl('mlp_direct_predictions', names(predictions_df)[i]) |
         # grepl('lstm_predictions', names(predictions_df)[i]) |
         # grepl('lstm_mimo_predictions', names(predictions_df)[i]) |
         # grepl('lstm_direct_predictions', names(predictions_df)[i]) |
         # grepl('nbeats_predictions', names(predictions_df)[i]) |
         # grepl('nbeats_mimo_predictions', names(predictions_df)[i]) |
         # grepl('nbeats_direct_predictions', names(predictions_df)[i]))
       {
        predictions_list[[names(predictions_df)[i]]] <- predictions_df[[names(predictions_df)[i]]]
      }
    }
    return(predictions_list)
  }
  
  predictions_df_multivariates <- predictions_df_mult(predictions_df) 
  return(predictions_df_multivariates)
}
predictions_res_multivariates <- load_predictions_res_mult()


# hybrid ml
predictions_multivariate_models_nonlinear <- function(predictions_df, i, nonlinear_multivariate_df)
{
  i_names <- strsplit(i, '_')[[1]]
  country <- i_names[length(i_names)-1]
  gender <- i_names[length(i_names)]
  country_gender <- paste(country, gender, sep='_')
  ### preciso tbm avaliar se devo utilizar a previsao dos residuos ou nao com base nos testes
  years_forecast <- 2010:2019
  # forecasting multivariado com hys-mf
  ## LC
  # get the start of kt and gc
  start_kt = start(arima_models[[country]][[paste0('lc_kt_forecast_', gender)]])[1]
  start_gc = start(arima_models[[country]][[paste0('rh_gc_forecast_', gender)]])[1]

  hys_mf_lc_kt <- ts(predictions_df[[i]][['lc_kt']], start=start_kt) + arima_models[[country]][[paste0("lc_kt_forecast_",gender)]]
  mx_hys_mf_lc_total_predict <- predict(lc_models_total[[country]], 
          years=years_forecast, 
          kt=hys_mf_lc_kt, type='rates')
    
  ## PLAT
  hys_mf_plat_kt1 <- ts(predictions_df[[i]][['plat_kt1']], start=start_kt) + arima_models[[country]][[paste0("plat_kt1_forecast_", gender)]]
  hys_mf_plat_kt2 <- ts(predictions_df[[i]][['plat_kt2']], start=start_kt) + arima_models[[country]][[paste0("plat_kt2_forecast_", gender)]]
  hys_mf_plat_gc <- ts(predictions_df[[i]][['plat_gc']], start=start_gc) + arima_models[[country]][[paste0("plat_gc_forecast_", gender)]]
  mx_hys_mf_plat_total_predict <- predict(object=plat_models_total[[country]],
                                        years=years_forecast,
                                        kt=rbind(hys_mf_plat_kt1,
                                                 hys_mf_plat_kt2),
                                        gc=c(na.omit(tail(plat_models_total[[country]]$gc, 101+2)), 
                                             hys_mf_plat_gc),
                                        type='rates')

  # stores in a list
  predictions_hys_multivariate_all <- list(
    'mx_hys_mf_lc_total_predict'=mx_hys_mf_lc_total_predict,
    'mx_hys_mf_plat_total_predict'=mx_hys_mf_plat_total_predict,
  )
  
  # write 
  names_split <- strsplit(i, '_')[[1]]
  model_name <- 'hys_mf'
  if (names_split[1] != 'predictions')
  {
    #TODO: adjust to consider single models 
    model_name <- paste(names_split[1:3], collapse = "_")
  } else if (names_split[3] != 'multivariate') 
  {
      model_name <- paste(names_split[3:5], collapse = '_')
  } 

  write.csv(mx_hys_mf_lc_total_predict, paste0('./results/predictions_', model_name,'_lc_',country_gender,'.csv'))
  write.csv(mx_hys_mf_plat_total_predict, paste0('./results/predictions_', model_name,'_plat_',country_gender,'.csv'))  
  
  return(predictions_hys_multivariate_all)
  
}

# single ml
predictions_multivariate_single_models <- function(predictions_df, i)
{
  
  i_names <- strsplit(i, '_')[[1]]
  country <- i_names[length(i_names)-1]
  gender <- i_names[length(i_names)]
  country_gender <- paste(country, gender, sep='_')
  years_forecast <- 2010:2019
  
  ## LC
  # get the start of kt and gc
  start_kt = start(arima_models[[country]][[paste0('lc_kt_forecast_', gender)]])[1]
  
  lc_kt_pred <- ts(predictions_df[[i]][['lc_kt']], start=start_kt)
  mx_lc_total_predict <- predict(lc_models_total[[country]], 
                                        years=years_forecast, 
                                        kt=lc_kt_pred, type='rates')
  
  ## PLAT
  plat_kt1_pred <- ts(predictions_df[[i]][['plat_kt1']], start=start_kt)
  plat_kt2_pred <- ts(predictions_df[[i]][['plat_kt2']], start=start_kt) 
  plat_gc_pred <- ts(predictions_df[[i]][['plat_gc']], start=start_gc)
  
  mx_plat_total_predict <- predict(object=plat_models_total[[country]],
                                          years=years_forecast,
                                          kt=rbind(plat_kt1_pred,
                                                   plat_kt2_pred),
                                          gc=c(na.omit(tail(plat_models_total[[country]]$gc, 101+2)), 
                                               plat_gc_pred),
                                          type='rates')

  # stores in a list
  predictions_single_multivariate_all <- list(
    'mx_lc_total_predict'=mx_lc_total_predict,
    'mx_plat_total_predict'=mx_plat_total_predict
  )
  
  # write 
  names_split <- strsplit(i, '_')[[1]]
  model_name <- paste(names_split[1], collapse = "_")
  if (names_split[3] == 'predictions'){
    model_name <- paste(names_split[1:2], collapse = "_")
  }

  write.csv(mx_lc_total_predict, paste0('./results/predictions_', model_name,'_lc_',country_gender,'.csv'))
  write.csv(mx_plat_total_predict, paste0('./results/predictions_', model_name,'_plat_',country_gender,'.csv'))

  
  return(predictions_single_multivariate_all)
  
}


write_all_predictions <- function(predictions_df)
{
  nonlinear_multivariate_df <- load_nonlinear_results(nonlinear_multivariate_json)
  predictions_ <- list()
  for(i in names(predictions_df))
  {
    cat(i)
    cat('\n')
    if (startsWith(i, 'mlp') | startsWith(i, 'lstm') | startsWith(i, 'nbeats')){
      predictions_[[i]] <- predictions_multivariate_single_models(predictions_df, i)
    } else {
      predictions_[[i]] <- predictions_multivariate_models_nonlinear(predictions_df, i, nonlinear_multivariate_df)
    }
  }
  return(predictions_)  
}

predictions_ <- write_all_predictions(predictions_res_multivariates)
