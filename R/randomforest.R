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

clf = randomForest(Class ~ ., data=train.all)
subset(as.data.frame(importance(clf)), MeanDecreaseGini < 0.02) # important features
varImpPlot(clf)
print(clf)
