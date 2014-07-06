#!/usr/bin/env R
# Toy example to generate a random forest over the dataset

library(randomForest)
library(miscTools)
library(plyr)
library(randomGLM)

# load data
train.fnc = read.csv('contest-data/train_FNC.csv')
train.labels = read.csv('contest-data/train_labels.csv')
train.sbm = read.csv('contest-data/train_SBM.csv')
train.labels <- factor(train.labels$Class)

# merge and clean up data
train.data = merge(train.fnc, train.sbm)
train.all = merge(train.data, train.labels)

#test.fnc = read.csv('contest-data/test_FNC.csv')
#test.sbm = read.csv('contest-data/test_SBM.csv')
#test.data = merge(test.fnc, test.sbm)

set.seed(0)


# predictions for testing data
test.fnc = read.csv('contest-data/test_FNC.csv')
test.sbm = read.csv('contest-data/test_SBM.csv')
test.data = merge(test.fnc, test.sbm)

RGLM = randomGLM(train.data, 
                 train.labels,
                 test.data,
                 classify=TRUE, keepModels=TRUE, nThreads=8)

predictedOOB = RGLM$predictedOOB
t=table(train.labels, predictedOOB)
print(t)
train.error = round((t[1,2]+t[2,1])/length(train.labels)*100, 4)
print(paste("train error rate = ", train.error))

predictions = as.data.frame(RGLM$predictedTest.response)
submission  = data.frame(Id=test.data$Id, Probability = predictions$"1")
write.csv(submission, file="predictions.csv")


# variable importance measures
varImp = RGLM$timesSelectedByForwardRegression

# Create a data frame that reports the variable importance measure of each feature.
datvarImp=data.frame(
  Feature=as.character(dimnames(RGLM$timesSelectedByForwardRegression)[[2]]),
  timesSelectedByForwardRegression= as.numeric(RGLM$timesSelectedByForwardRegression))

#Report the 20 most significant features
top20features = datvarImp[rank(-datvarImp[,2],ties.method="first")<=20,]
print(top20features)
