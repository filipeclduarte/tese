library(ggeffects)
library(DHARMa)
library(lme4)
library(ggplot2)
library(dplyr)
library(jtools) # summarizing and visualizing reg models
library(pbkrtest) # testes 

# read dataset
df <- read.csv('df_reg.csv', header = T)
df_mase <- read.csv('df_reg_mase.csv', header = T)
str(df)
df$model = as.factor(df$model)
df_mase$model = as.factor(df_mase$model)
str(df)
df <- within(df, model <- relevel(model, ref = 'lc'))
df_mase <- within(df_mase, model <- relevel(model, ref = 'lc'))
df$year <- df$year - min(df$year)
df_mase$year <- df_mase$year - min(df_mase$year)

df_mase <- df_mase %>% 
  filter(model != "proposed_val")

###### Colocar o LC como padrão e comparar todos os modelos com ele
lm_mod <- glm('value ~ model + country + gender + year ', 
             data=df, family=gaussian(link='log'))
gam_mod <- glm('value ~ model + country + gender + year ', 
                   data=df,
                   family=Gamma(link='log'))
summary(lm_mod)
summary(gam_mod)
summ(lm_mod, robust='HC3', confint=T, digits=3, exp=T)
exp(coef(gam_mod))
exp(coef(gam_mod))-1
write.csv(exp(coef(gam_mod))-1, 'gam_mod_coef.csv')

gam_mod_mase <- glm('value ~ model + country + gender + year ', 
               data=df_mase,
               family=Gamma(link='log'))
summary(gam_mod_mase)
write.csv(exp(coef(gam_mod_mase))-1, 'gam_mod_coef_mase.csv')
exp(coef(gam_mod_mase))
exp(coef(gam_mod_mase))-1


summ(gam_mod, robust='HC3', confint=T, digits=3, exp=T)
AIC(gam_mod)
BIC(gam_mod)
testUniformity(gam_mod)# tests if the overall distribution conforms to expectations
testDispersion(gam_mod) #tests if the simulated dispersion is equal to the observed dispersion
testQuantiles(gam_mod) # fits a quantile regression or residuals against a predictor (default predicted value), and tests of this conforms to the expected quantile
plot(y=standardize(residuals(gam_mod)), x=fitted.values(gam_mod))
simulationOutput <- simulateResiduals(fittedModel = gam_mod, n = 500, use.u = T)
plot(simulationOutput)
gam_mod$deviance/gam_mod$df.residual
plot(gam_mod)
hist(residuals(gam_mod))
summary(gam_mod)$deviance
summary(gam_mod) # residual deviance = 901. DF = 8579 df
summary(gam_mod)$deviance/0.1029516 # 8751,67 que é próximo do df
# residual deviance Test
1-pchisq(deviance(gam_mod)/summary(gam_mod)$disp, gam_mod$df.resid) # p-value = 0.094
anova(gam_mod, test='F')

plot(ggpredict(gam_mod, terms = "year"), rawdata = TRUE) +
  ggplot2::ggtitle("Predicted values of totalvalue: gamma model")


plot(gam_mod$y, fitted(gam_mod))
abline(0,1, col='red')
plot(fitted(gam_mod), residuals(gam_mod))
summ(gam_mod, confint=T, digits=3, exp=T)
plot_summs(gam_mod, robust='HC3', plot.distributions = T)
barplot(sort(exp(coef(gam_mod))-1), horiz = T) # ver como faz bar plot dos coefs
