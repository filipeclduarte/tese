library(readr)
library(demography)
source('hmd_mx.R')

# french: 1950 - 2019
french_mortality_data <- hmd_mx("FRATNP", "email", "password")
write.csv(as.data.frame(log(t(french_mortality_data$rate$total))), 'data/french_total.csv', row.names = T)
write.csv(as.data.frame(log(t(french_mortality_data$rate$female))), 'data/french_female.csv', row.names = T)
write.csv(as.data.frame(log(t(french_mortality_data$rate$male))), 'data/french_male.csv', row.names = T)

# australia: 1921 - 2019
australia_mortality_data <- hmd_mx("AUS", "email", "password")
write.csv(as.data.frame(log(t(australia_mortality_data$rate$total))), 'data/australia_total.csv', row.names = T)
write.csv(as.data.frame(log(t(australia_mortality_data$rate$female))), 'data/australia_female.csv', row.names = T)
write.csv(as.data.frame(log(t(australia_mortality_data$rate$male))), 'data/australia_male.csv', row.names = T)

# japao: 1950 - 2019
japao_mortality_data <- hmd_mx("JPN", "email", "password")
write.csv(as.data.frame(log(t(japao_mortality_data$rate$total))), 'data/japao_total.csv', row.names = T)
write.csv(as.data.frame(log(t(japao_mortality_data$rate$female))), 'data/japao_female.csv', row.names = T)
write.csv(as.data.frame(log(t(japao_mortality_data$rate$male))), 'data/japao_male.csv', row.names = T)

# portugal: 1950 - 2019
portugal_mortality_data <- hmd_mx("PRT","email", "password")
write.csv(as.data.frame(log(t(portugal_mortality_data$rate$total))), 'data/portugal_total.csv', row.names = T)
write.csv(as.data.frame(log(t(portugal_mortality_data$rate$female))), 'data/portugal_female.csv', row.names = T)
write.csv(as.data.frame(log(t(portugal_mortality_data$rate$male))), 'data/portugal_male.csv', row.names = T)

# EUA: 1950 - 2019
eua_mortality_data <- hmd_mx("USA", "email", "password")
write.csv(as.data.frame(log(t(eua_mortality_data$rate$total))), 'data/eua_total.csv', row.names = T)
write.csv(as.data.frame(log(t(eua_mortality_data$rate$female))), 'data/eua_female.csv', row.names = T)
write.csv(as.data.frame(log(t(eua_mortality_data$rate$male))), 'data/eua_male.csv', row.names = T)

