library(readr)
library(forecast)
library(demography)


load_predictions <- function()
{
  
  # Lista todos os arquivos CSV na pasta "results" que contem a palavra "predictions"
  arquivos <- list.files(path='results',pattern = "predictions.*\\.csv$")
  #TODO: preciso ler arquivos dos singles
  
  # Cria uma lista vazia para armazenar os dataframes
  lista_df <- list()
  
  # Usa um loop para ler cada arquivo CSV e salva-lo como um elemento da lista
  for (i in 1:length(arquivos)) {
    nome_arquivo <- paste0("results/", arquivos[i])
    chave_lista <- gsub("\\.csv$", "", nome_arquivo)
    df <- read.csv(nome_arquivo)
    chave_lista <- gsub("results/", "", arquivos[i])
    chave_lista <- gsub("\\.csv", "", chave_lista)
    # lista_df[[chave_lista]] <- df
    # posso fazer isso pq so quero os multivariados
    rownames(df) <- df$x
    lista_df[[chave_lista]] <- df[,2:length(df)]
  }
  return(lista_df)
}