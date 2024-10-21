library(readr)
library(forecast)

portugal_male <- read_csv('data/portugal_male.csv')
portugal_male_ <- portugal_male[,c('X1', '0':'100')]
portugal_male_[sapply(portugal_male_, is.infinite)] <- NA
portugal_male_[, '100'] <- na.interp(portugal_male_[, '100']) # interpolation
portugal_male <- cbind(portugal_male_, portugal_male[,c('101':'109', '110+')])
colnames(portugal_male)[1] <- ''
row.names(portugal_male) <- portugal_male[,1]
write.csv(portugal_male[, -1], 'data/portugal_male.csv')     
