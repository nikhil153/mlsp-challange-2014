#!/usr/bin/env R
# Toy example to generate a random forest over the dataset
# inspiration: 
# - http://blog.yhathq.com/posts/comparing-random-forests-in-python-and-r.html

library(randomForest)
library(miscTools)
library(plyr)

# load data
train.fnc = read.csv('contest-data/train_FNC.csv')
train.labels = read.csv('contest-data/train_labels.csv')
train.sbm = read.csv('contest-data/train_SBM.csv')
train.labels$Class <- factor(train.labels$Class)

# merge and clean up data
train.data = merge(train.fnc, train.sbm)
train.all = merge(train.data, train.labels)

train.fnc$Id <- NULL   # assume sorted
train.labels$Id <- NULL
train.sbm$Id <- NULL
train.data$Id <- NULL
train.all$Id <- NULL

set.seed(0)

clf = randomForest(Class ~ ., data=train.all, ntree=10000)
subset(as.data.frame(importance(clf)), MeanDecreaseGini < 0.02) # important features
varImpPlot(clf)
print(clf)

# predictions for testing data
test.fnc = read.csv('contest-data/test_FNC.csv')
test.sbm = read.csv('contest-data/test_SBM.csv')
test.data = merge(test.fnc, test.sbm)

# predict on dataset (with Id column removed)
predictions = as.data.frame(predict(clf, test.data[,2:ncol(test.data)], type="prob"))
submission = data.frame(Id=test.data$Id, Probability = predictions$"1")
write.csv(submission, file="predictions.csv")