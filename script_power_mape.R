rm(list=ls())

pckgs <- c('readr','tidyr','dplyr','magrittr',
           'ggplot2','cowplot')
for (pck in pckgs) { library(pck,character.only = T)}

##############################
# ---- FUNCTION SUPPORT ---- #

find_dir_cell <- function() {
  dir_base <- getwd()
  cpu <- as.character(Sys.info()["nodename"])
  os <- as.character(Sys.info()['sysname'])
  if (cpu == 'RT5362WL-GGB') {
    stopifnot(os == 'Windows')
    dir_cell <- 'D:\\projects\\GICell'
  } else if (cpu == 'snowqueen') {
    stopifnot(os == 'Linux')
    dir_cell <- file.path(dir_base, '..')
  } else {
    stopifnot(F)
  }
  return(dir_cell)
}

mae_fun <- function(act, pred) {
  aerr <- abs(act - pred)
  return( apply(aerr, 2, mean) )
}

mape_fun <- function(act, pred) {
  aerr <- abs(act - pred)
  return( apply(aerr / act, 2, mean) )
}

r2_fun <- function(act, pred) {
  err <- act - pred
  return(1 - apply(err,2,var)/apply(act,2,var))
}

dgp_r2 <- function(n, r2, k) {
  x <- matrix(rnorm(n * k),ncol=k)
  sig2 <- 1/r2 - 1
  z <- sqrt(sig2) * matrix(rnorm(n*k),ncol=k)
  y <- x + z
  return(list(y=y, x=x))
}

############################
# ---- LOAD CELL DIST ---- #

dir_base <- find_dir_cell()
dir_output <- file.path(dir_base, 'output')
dir_figures <- file.path(dir_output, 'figures')

df_cells <- read_csv(file.path(dir_output,'df_cells.csv'),
                     col_types = cols(id=col_character(),.default = col_integer()))
# Calculate inflammatory
df_cells <- df_cells %>% 
  mutate(inflam=eosinophil+neutrophil+plasma+lymphocyte) %>% 
  dplyr::rename(eosin=eosinophil) %>% 
  dplyr::select(id,eosin,inflam)

df_cells %>% pivot_longer(!id,names_to='cell') %>% 
  ggplot(aes(x=log(value),fill=cell)) + theme_bw() + 
  geom_density(color='black',alpha=0.5) + 
  labs(x='Log(# Cells)',y='Density',subtitle='Exclude 0 counts') + 
  ggtitle('Cell counts are roughly log-normal') + 
  theme(legend.position=c(0.2,0.85)) + 
  scale_fill_discrete(name='Cell type',labels=c('Eosinophil','Inflammatory'))



##############################
# ---- VARIATION IN MAE ---- #


# Relationship between R2 and MAPE
r2_seq <- seq(0.1,0.9,0.01)
for (r2 in r2_seq) {
  tmp_yx <- dgp_r2(n=1000,r2=r2,k=100)
  X <- tmp_yx$x
  Y <- tmp_yx$y
  
}

hist(abs(Y[,2] - X[,2])/Y[,2])

mape_fun(Y,X)


mean(r2_fun(Y,X))








