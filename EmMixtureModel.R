library(mvtnorm)
library(RUnit)
library(mclust)
library(MASS)

##################################################################
###                 Em                                         ###
##################################################################

#############################
###    Posteriori         ###
#############################
  

posteriori <- function(data, k, K, pi, mu, sigma) {
  a <- pi[[k]] * dmvnorm(data, mu[[k]], sigma[[k]])
  b <- sum(sapply(1:K, function(j) { pi[[j]] * dmvnorm(data, mu[[j]], sigma[[j]]) }))
  a / b
}

#############################
###    E-step             ###
#############################

Estep <- function(data, pi, mu, sigma) {
  K <- length(mu)
  apply(data, 1, function(x) {
    sapply(1:K, function(k) {
      posteriori(x, k, K, pi, mu, sigma)
    })
  })
}

#############################
###    M-Step             ###
#############################

Mstep <- function(data, K, gammaKn) {
  N <- nrow(data)
  M <- ncol(data)
  
  nKList <- lapply(1:K, function(k)
    sum(gammaKn[k, ]))
  
  piNext <- lapply(1:K, function(k)
    nKList[[k]] / N)
  
  muNext <- lapply(1:K, function(k) {
    sum <- rowSums(sapply(1:N, function(j)
      gammaKn[k, j] * data[j,]))
    sum / nKList[[k]]
  })
  
  sigmaNext <- lapply(1:K, function(k) {
    sum <-
      rowSums(sapply(1:N, function(j)
        gammaKn[k, j] * ((data[j, ] - muNext[[k]]) %*% t(data[j, ] - muNext[[k]]))))
    matrix(sum / nKList[[k]], ncol = M)
  })
  
  list(piNext, muNext, sigmaNext)
}

################################
### good parameters for init ###
################################

initialize <- function(data, K, initNumber){
  
  N <- nrow(data)
  M <- ncol(data)
  BicValue <- +Inf
  
  for (q in 1:initNumber) {
    result <- kmeans(data, K)
    candidate_pi <- c(result$size/N)
    candidate_mu <- list()
    for (k in 1:K) {
      for (i in 1:M) {
        candidate_mu[[k]] <- apply(data[result$cluster==k,],2, mean)
      }
    }
    candidate_sigma <- list()
    for(k in 1:K){
      candidate_sigma[[k]] <- cov(data[result$cluster==k,])
    }
    candidateBicValue <- -2 * sum(apply(data, 1, "+", log(sum(
      sapply(1:K, function(k)
        candidate_pi[k] * dmvnorm(data, candidate_mu[[k]], candidate_sigma[[k]]))
    )))) + ((K * ((M * (M + 1) / 2) + M + 1)) - 1) * log(N)
    
    if(candidateBicValue < BicValue){
      cat("i found a better init parameters\n")
      BicValue <- candidateBicValue
      mu <- candidate_mu
      pi <- candidate_pi
      sigma <- candidate_sigma
    }
  }
  return(list(pi,mu,sigma,BicValue))
}

#################################
### EMcluster - main function ###
#################################

emcluster <- function(data, K, initNumber){
  
  initValues <- initialize(data, K, initNumber)
  start_pi <- initValues[[1]]
  start_mu <- initValues[[2]]
  start_sigma <- initValues[[3]]

  gammaKn = Estep(data, start_pi, start_mu, start_sigma)
  v <- Mstep(data, K, gammaKn)
  
  old.log.like <- -Inf
  for(n in 1:200) {
    gammaKn = Estep(data, v[[1]], v[[2]], v[[3]])
    v <- Mstep(data, K, gammaKn)
    cat(sprintf("n=%d pi=%s\n", n, v[[1]]))
    log.like <-
      sum(apply(data, 1, "+", log(sum(
        sapply(1:K, function(k)
          v[[1]][[k]] * dmvnorm(data, v[[2]][[k]], v[[3]][[k]]))
      ))))
    
    if(log.like - old.log.like < 0.001){
      break
    }else{
      old.log.like <- log.like
    }
  }
  
  M <- ncol(data)
  N <- nrow(data)
  bic <- -2 * log.like + ((M * ( (M * (M+1) / 2) + M + 1) )- 1) * log(N)
  partition <- apply(t(gammaKn),1,which.max)
  clusters <- c()
  for(k in 1:K){
    clusters[k] <- sum(partition == k) 
  }
  return(list(v, t(gammaKn), partition, clusters, bic))
}

###################################################################### 
### Mnist                                                          ###
######################################################################

load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename, 'rb')
    readBin(f,
            'integer',
            n = 1,
            size = 4,
            endian = 'big')
    ret$n = readBin(f,
                    'integer',
                    n = 1,
                    size = 4,
                    endian = 'big')
    nrow = readBin(f,
                   'integer',
                   n = 1,
                   size = 4,
                   endian = 'big')
    ncol = readBin(f,
                   'integer',
                   n = 1,
                   size = 4,
                   endian = 'big')
    x = readBin(
      f,
      'integer',
      n = ret$n * nrow * ncol,
      size = 1,
      signed = F
    )
    ret$x = matrix(x, ncol = nrow * ncol, byrow = T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename, 'rb')
    readBin(f,
            'integer',
            n = 1,
            size = 4,
            endian = 'big')
    n = readBin(f,
                'integer',
                n = 1,
                size = 4,
                endian = 'big')
    y = readBin(f,
                'integer',
                n = n,
                size = 1,
                signed = F)
    close(f)
    y
  }
  train <<- load_image_file('train-images-idx3-ubyte')
  test <<- load_image_file('t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('train-labels-idx1-ubyte')
  test$y <<- load_label_file('t10k-labels-idx1-ubyte')
}


show_digit <- function(x,
                       col = gray(12:1 / 12),
                       title = 'image') {
  image(matrix(x, nrow = 28)[, 28:1], col = col, main = title)
}


load_mnist()


################################################################
### ACP sur mnist                                            ###
################################################################

mnist_pca <-
  prcomp(train$x,
         retx = TRUE,
         center = TRUE,
         scale. = FALSE)
dataMnist <-  mnist_pca$x[1:1000, 1:10]


################################################################
### Application sur iris                                     ###
################################################################

datairis <- as.matrix(iris[, c(1, 2, 3, 4)])

res_em_iris <- emcluster(datairis, 3, 1000)
#les paramètres estimés
res_em_iris[[1]]
#la matrice des posterioris
res_em_iris[[2]]
#la partition estimé avec maximum des posterioris
res_em_iris[[3]]
#les clusters
res_em_iris[[4]]
res_em_iris_cluster <- sort(res_em_iris[[4]])
#la valeur du bic
res_em_iris[[5]]


res_mclust_iris <- Mclust(datairis, 3)
#les paramètres estimés
res_mclust_iris$parameters
#la partition estimé avec maximum des posterioris
res_mclust_iris$classification
#la valeur du bic
res_mclust_iris$bic


#plot partition estimé 
K <- 3
layout(matrix(c(1, 2), 2, 2, byrow = TRUE))
plot(
  seq(1:3),
  c(
    res_em_iris_cluster[1],
    res_em_iris_cluster[2],
    res_em_iris_cluster[3]
  ),
  type = 'h',
  lwd = 10,
  xlab = "Cluster",
  ylab = "Nombre d'individus dans le cluster",
  main =  "Notre Em-algo, partition estimé avec 3 clsuters"
)

clusters <- c()
for (k in 1:K) {
  clusters[k] <- sum(res_mclust_iris$classification == k)
}

res_mclust_iris_cluster <- sort(clusters)

plot(
  seq(1:3),
  c(
    res_mclust_iris_cluster[1],
    res_mclust_iris_cluster[2],
    res_mclust_iris_cluster[3]
  ),
  type = 'h',
  lwd = 10,
  xlab = "Cluster",
  ylab = "Nombre d'individus dans le cluster",
  main =  "Mclust, partition estimé avec 3 clusters"
)

################################################################
### Application sur Mnist                                    ###
################################################################

res_em_mnist <- emcluster(dataMnist, 10, 1000)
#les paramètres estimés
res_em_mnist[[1]]
#la matrice des posterioris
res_em_mnist[[2]]
#la partition estimé avec maximum des posterioris
res_em_mnist[[3]]
#les clusters
res_em_mnist[[4]]
res_em_mnist_cluster <- sort(res_em_mnist[[4]])
#la valeur du bic
res_em_mnist[[5]]


res_mclust_mnist <- Mclust(dataMnist, 10)
#les paramètres estimés
res_mclust_mnist$parameters
#la partition estimé avec maximum des posterioris
res_mclust_mnist$classification
#la valeur du bic
res_mclust_mnist$bic


#plot partition estimé 
K <- 10

layout(matrix(c(1, 2), 2, 2, byrow = TRUE))
plot(
  seq(1:K),
  c(
    res_em_mnist_cluster[1],
    res_em_mnist_cluster[2],
    res_em_mnist_cluster[3],
    res_em_mnist_cluster[4],
    res_em_mnist_cluster[5],
    res_em_mnist_cluster[6],
    res_em_mnist_cluster[7],
    res_em_mnist_cluster[8],
    res_em_mnist_cluster[9],
    res_em_mnist_cluster[10]
    
  ),
  type = 'h',
  lwd = 10,
  xlab = "Cluster",
  ylab = "Nombre d'individus dans le cluster",
  main =  "Notre Em-algo, partition estimé avec 10 clsuters"
)

clusters <- c()
for (k in 1:K) {
  clusters[k] <- sum(res_mclust_mnist$classification == k)
}

res_mclust_mnist_cluster <- sort(clusters)

plot(
  seq(1:K),
  c(
    res_mclust_mnist_cluster[1],
    res_mclust_mnist_cluster[2],
    res_mclust_mnist_cluster[3],
    res_mclust_mnist_cluster[4],
    res_mclust_mnist_cluster[5],
    res_mclust_mnist_cluster[6],
    res_mclust_mnist_cluster[7],
    res_mclust_mnist_cluster[8],
    res_mclust_mnist_cluster[9],
    res_mclust_mnist_cluster[10]
  ),
  type = 'h',
  lwd = 10,
  xlab = "Cluster",
  ylab = "Nombre d'individus dans le cluster",
  main =  "Mclust, partition estimé avec 10 clusters"
)
