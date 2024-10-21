### Teste nemenyi e nemenyi MCB 
library(tsutils)
library(dplyr)
library(tidyr)


#### ERROR ALL
df_error_abs <- read.csv('error_all.csv')
df_error_abs <- df_error_abs[,-c(1)]
str(df_error_abs)

df_error_abs <- df_error_abs %>% 
  pivot_longer(cols="X2010":"X2019", names_to='year', values_to='valor') %>% 
  pivot_wider(names_from = 'model', values_from='valor')

# posso fazer o teste para cada country_gender
df_error_abs[, 'diff'] =df_error_abs[, 'proposto'] - df_error_abs[, 'arima'] 
cg = 'eua_male'

median(df_error_abs[df_error_abs$country_gender==cg,]$proposto)
median(df_error_abs[df_error_abs$country_gender==cg,]$arima)
median(df_error_abs[df_error_abs$country_gender==cg,]$diff)
mean(df_error_abs[df_error_abs$country_gender==cg,]$diff)
hist(df_error_abs[df_error_abs$country_gender==cg,]$diff / 
       sd(df_error_abs[df_error_abs$country_gender==cg,]$diff))
t.test(df_error_abs[df_error_abs$country_gender==cg,]$diff)
wilcox.test(df_error_abs[df_error_abs$country_gender==cg,]$proposto,
            df_error_abs[df_error_abs$country_gender==cg,]$arima,
            paired=T, alternative="less")
var.test(df_error_abs[df_error_abs$country_gender==cg,]$proposto,
         df_error_abs[df_error_abs$country_gender==cg,]$arima)
library(intervcomp)
Bonett.Seier.test(df_error_abs[df_error_abs$country_gender==cg,]$proposto,
                  df_error_abs[df_error_abs$country_gender==cg,]$arima, alternative = 'two.sided')

df_error_f <- df_error_abs %>% 
  select('arima', 
         'proposto',
         'arima_lstm',  'arima_mlp', 
         'mlp_mimo', 'mlp_direct',
         'lc', 'plat', 
         -country_gender, -year, -age) %>% 
  rename_with(stringr::str_replace,
              pattern = "predictions_", replacement = "") %>% 
  rename_with(stringr::str_replace,
              pattern = "_", replacement = "-") %>%
  rename_with(stringr::str_replace,
              pattern = "_", replacement = "-") %>%
    rename_all(toupper)

apply(t(data.frame(apply(df_error_abs_2010, 1, rank))), FUN = mean, MARGIN=2)

t <- nemenyi(df_error_f, plottype='vmcb')
t$means
t$intervals
# nemenyi(df_error_abs_2010, plottype = 'vmcb')
t <- nemenyi(df_error_abs %>% select(-country_gender, -age, -year), plottype = 'vmcb')
# nemenyi(df_error_abs %>% select(-country_gender, -age, -year))
t$means
t$intervals

