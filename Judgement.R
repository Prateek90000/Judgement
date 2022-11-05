rm(list =ls())

#install.packages("dplyr")
#install.packages("magrittr")
#install.packages("ggplot2")
#install.packages("caTools")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("ROCR")
#install.packages("randomForest")
#install.packages("caret")
#install.packages("e1071")

library("dplyr")
library("magrittr")
library("ggplot2")
library("caTools")
library("rpart")
library("rpart.plot")
library("ROCR")
library("randomForest")
library("caret")
library("e1071")

judgements = read.csv("judgements.csv", stringsAsFactors = TRUE)

judgements_main = judgements[,-c(1,2)]

model_logreg = glm(Reverse ~ . , data = judgements_main,family = binomial)
summary(model_logreg)

# create a tree model
set.seed(7)
split1 = sample.split(judgements_main$Reverse,SplitRatio = 0.8)

train = (subset(judgements_main,split1 == TRUE))
test = (subset(judgements_main,split1 == FALSE))

# check logistic model for train subset
modelTrain_logreg = glm(Reverse ~ . , data = train,family = binomial)
summary(modelTrain_logreg)

preds = predict(modelTrain_logreg,type = "response",newdata = test)
table(test$Reverse,preds>0.5)
LM_accuracy =(31+48)/(31+20+14+48)

truthTable = table(test$Reverse,preds>0.5)
LM_AutomaticAccuracy = (truthTable[1,1]+truthTable[2,2])/(truthTable[1,1]+truthTable[1,2]+truthTable[2,1]+truthTable[2,2])


LM_FalsePositive = (truthTable[1,2])/(truthTable[1,2]+truthTable[1,1])
LM_FalseNegative = (truthTable[2,1])/(truthTable[2,1]+truthTable[2,2])

#sensitivity
LM_sensitivity = (truthTable[2,2])/(truthTable[2,1]+truthTable[2,2])
LM_TRUEPOSITIVE = (truthTable[2,2])/(truthTable[2,1]+truthTable[2,2])
#specificity
LM_specificity = (truthTable[1,1])/(truthTable[1,1]+truthTable[1,2])
LM_TRUENEGATIVE = (truthTable[1,1])/(truthTable[1,1]+truthTable[1,2])

#Finally Tree Model
# CART
treeModel = rpart(Reverse ~ ., data = train,method = "class",minbucket = 5)
# method - class ---- this gives a classification tree, minbucket = minimum size of bucket
prp(treeModel)
predsOfTree = predict(treeModel, newdata = test, type = "class")
treeTruthTable = table(test$Reverse,predsOfTree)
treeAccuracy = (treeTruthTable[1,1]+treeTruthTable[2,2])/(treeTruthTable[1,1]+treeTruthTable[1,2]+treeTruthTable[2,1]+treeTruthTable[2,2])
treeTP = (treeTruthTable[2,2])/(treeTruthTable[2,2]+treeTruthTable[2,1])
treeTN = (treeTruthTable[1,1])/(treeTruthTable[1,1]+treeTruthTable[1,2])
treeFP = (treeTruthTable[1,2])/(treeTruthTable[1,1]+treeTruthTable[1,2])
treeFN = (treeTruthTable[2,1])/(treeTruthTable[2,2]+treeTruthTable[2,1])

# Get the prediction curve
# ROC curve = characteristic curve (operator characteristic)
predsOfTreeROC = predict(treeModel, newdata = test)


curve = prediction(predsOfTreeROC[,2], test$Reverse)
# performanceCurve = performance(curve,"tru_positive_rate","false_positive_rate")
performanceCurve = performance(curve,"tpr","fpr")
plot(performanceCurve)

auc <- as.numeric(performance(curve, "auc")@y.values)

performanceCurve2 = performance(curve,"fnr","fpr")
plot(performanceCurve2)

# Random Forest

test$Reverse = as.factor(test$Reverse)
train$Reverse = as.factor(train$Reverse)
set.seed(5)
modelRandomForest = randomForest(Reverse ~.,data=train, ntree=100, nodesize=5)
randomForestPreds = predict(modelRandomForest, newdata = test)
rfTable = table(test$Reverse, randomForestPreds)

rfAccuracy = (rfTable[1,1]+rfTable[2,2])/(rfTable[1,1]+rfTable[1,2]+rfTable[2,1]+rfTable[2,2])
rfTP = (rfTable[2,2])/(rfTable[2,2]+rfTable[2,1])
rfTN = (rfTable[1,1])/(rfTable[1,1]+rfTable[1,2])
rfFP = (rfTable[1,2])/(rfTable[1,1]+rfTable[1,2])
rfFN = (rfTable[2,1])/(rfTable[2,2]+rfTable[2,1])