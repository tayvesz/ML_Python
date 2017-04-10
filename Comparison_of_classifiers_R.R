setwd('/home/yves/tel/Aggregation/')
rm(list=ls())
library(MASS)
library(class)
library(robustbase)
library(neuralnet)
library(gbm)
library(randomForest)

Normal2 <- function(numLearn, numTest){
  library("MASS")
  l1 <- mvrnorm(numLearn, c(0,0), matrix(c(1,1,1,4), nrow = 2, ncol = 2, byrow=TRUE))
  l2 <- mvrnorm(numLearn, c(1,1), matrix(c(4,4,4,16), nrow = 2, ncol = 2, byrow = TRUE))
  t1 <- mvrnorm(numTest, c(0,0), matrix(c(1,1,1,4), nrow = 2, ncol = 2, byrow=TRUE))
  t2 <- mvrnorm(numTest, c(1,1), matrix(c(4,4,4,16), nrow = 2, ncol = 2, byrow = TRUE))
  learnData <- rbind(cbind(l1, rep(0, numLearn)),cbind(l2, rep(1, numLearn)))
  testData <- rbind(cbind(t1, rep(0, numTest)),cbind(t2, rep(1, numTest)))
  rez <- list(learn = learnData, test = testData)
  return(rez)
}

error.lda <- NULL
error.qda <- NULL
error.qdarob <- NULL
error.knn <- NULL
error.nn <- NULL
error.boost <- NULL
error.rf <- NULL
error.clevernn <- NULL



M <- 100
# Run the simulation
set.seed(1)
for (i in 1:M){
  # Generate data
  simdata <- Normal2(200, 500) # 1
  
  data.train <- data.frame(simdata$learn)
  data.test <- data.frame(simdata$test)
  
  # Train classifiers
  machine.lda <- lda(X3 ~ ., data.train)
  machine.nn <- neuralnet(X3 ~ X1 + X2, data.train, hidden = 0, err.fct = "sse", act.fct = "logistic")
  data.train[,"X3"] <- as.factor(data.train[,"X3"])   # because target variable need to be a factor
  machine.rf <- randomForest(X3 ~ ., data = data.train, ntree = 100)
  
  # Calculate classification error
  error.lda <- c(error.lda, mean(predict(machine.lda, data.test[,1:2])$class != data.test[,3]))
  if (!is.null(machine.nn$net.result)){
    error.nn <- c(error.nn, mean(as.numeric(compute(machine.nn, data.test[,1:2])$net.result>0.5) !=  data.test[,3]))
  }
  if (!is.null(machine.nn$net.result)){
    error.nn <- c(error.nn, mean(as.numeric(compute(machine.nn, data.test[,1:2])$net.result>0.5) !=  data.test[,3]))
  }
  
  error.rf <- c(error.rf, mean(predict(machine.rf, data.test[,1:2]) != data.test[,3]))
  cat(i, " ", sep = "")
}
cat("\n")


# Plot the errors
errors <- list(error.lda, error.rf, error.nn)
names <- c("lda", "rf", "nn")
boxplot(errors, names = names, horizontal = TRUE, main = "Misclassification rate")

# in our context random forest provided some better result. 
# Actually,  the best classifier for this type of data is QDA.
# a single neural network is not enough good to provide a good classification in this case
