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
train.graph   = read.csv('contest-data/train_FNC_graph.csv')
train.labels$Class <- factor(train.labels$Class)

# merge and clean up data
train.data    = merge(train.fnc, train.sbm)
train.data    = merge(train.data, train.graph)
train.all     = merge(train.data, train.labels)

set.seed(0)

# first pass with all the data
clf = randomForest(Class ~ ., data=train.all, ntree=10000)

# second pass with just the important data
gini_threshold = 0.1
features = row.names(subset(imp, MeanDecreaseGini > gini_threshold))
clf = randomForest(Class ~ ., data=train.all[, append(features, "Class")], ntree=10000)

# predictions for testing data
test.fnc = read.csv('contest-data/test_FNC.csv')
test.sbm = read.csv('contest-data/test_SBM.csv')
test.graph   = read.csv('contest-data/test_FNC_graph.csv')
test.data = merge(test.fnc, test.sbm)
test.data = merge(test.data, test.graph)

# predict on dataset
predictions = as.data.frame(predict(clf, test.data[,append(features, "Class")], type="prob"))
submission = data.frame(Id=test.data$Id, Probability = predictions$"1")
write.csv(submission, file="predictions.csv")

#clf = randomForest(Class ~ ., data=train.all, ntree=10000)
#imp = as.data.frame(importance(clf))
#for (gini_threshold in seq(0.075,0.15,0.01)) { 
#	features = row.names(subset(imp, MeanDecreaseGini > gini_threshold))
#	clf = randomForest(Class ~ ., data=train.all[, append(features, "Class")], ntree=10000)
#	print(gini_threshold)
#	print(clf)
#}


