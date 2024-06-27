library(caret)
library(ggplot2)
library(pROC)

dataset = read.csv("D:\\Statistica Applicata\\Titanic\\train.csv")
summary(dataset)
head(dataset)

# Data cleaning
dataset_Id = dataset$PassengerId

idx = which(names(dataset) %in% c("Name", "Ticket", "Cabin", "PassengerId"))
dataset = dataset[,-idx]
summary(dataset)
dataset$Sex = as.factor(dataset$Sex)
dataset$Survived = as.factor(dataset$Survived)
dataset$Pclass = as.factor(dataset$Pclass)
dataset$Embarked = as.factor(dataset$Embarked)
summary(dataset)

mean_age = mean(dataset$Age[which(!is.na(dataset$Age))])
dataset$Age[which(is.na(dataset$Age))] = mean_age
summary(dataset)


featurePlot(x = dataset[, 1:(ncol(dataset)-1)], 
            y = dataset$Survived, 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

dv = dummyVars(Survived ~ ., data = dataset)
ds = predict(dv, newdata = dataset)
summary(ds)
# preProcessing

# There are neither zero-variance, nor near-zero variance variables
nzv = nearZeroVar(dataset, saveMetrics = TRUE)
centerScale = preProcess(dataset, method = c("center", "scale"))

# data splitting
idx = createDataPartition(dataset$Survived, p=0.85, list=FALSE)
trainset = predict(centerScale, dataset[idx,])
testset = predict(centerScale, dataset[-idx,])
summary(trainset)



# parallelism
library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

levels(trainset$Survived) = c("c1","c2")
levels(testset$Survived) = c("c1","c2")
# model tuning
control = trainControl(method="cv", number=10, classProbs=T, savePredictions = TRUE)
model.svm = train(Survived~., data = trainset, method="svmRadial", tuneLength = 10, trControl = control, importance = TRUE)

# feature selection
varImp(model.svm)

# confusion matrix
predictions = predict(model.svm, testset)
sink(file="D:\\Statistica Applicata\\caret\\images\\file.txt")
confusionMatrix(model.svm)
sink(file=NULL)
ggplot(model.svm)

# ROC curve
library(pROC)
predictions_prob = predict(model.svm, testset)
rc = roc(testset[,"Survived"], predictions_prob$c1)
coords(rc, "best", ret = "threshold")
plot.roc(testset[,"Survived"], predictions_prob$c1)


control = trainControl(method="cv", number=10)
model.bayes = train(Survived~., data = trainset, method="naive_bayes", tuneLength = 3, trControl = control, importance = TRUE)
varImp(model.bayes)
plot(model.bayes)



confusionMatrix(model.bayes)

control = trainControl(method="cv", number=10)
grid = expand.grid(mtry = c(sqrt(ncol(trainset)), ncol(trainset)/3), min.node.size = c(1, 5), splitrule="gini")
model.rf = train(Survived~., data = trainset, method="ranger", trControl = control, importance = "impurity", tuneGrid = grid)
varImp(model.rf)
plot(model.rf)

confusionMatrix(model.rf)

# deregister parallelism
registerDoSEQ()
stopCluster(cl)
