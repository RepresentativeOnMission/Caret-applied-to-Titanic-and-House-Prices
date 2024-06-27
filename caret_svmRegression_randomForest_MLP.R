# INTRODUCTION
# In this work we predict the price of houses using 3 models: Regression SVM, Random Forest and MLP. We then proceed 
# to compare these three models on the validation set, using R^2 and adjusted R^2 functions, in order to pick the best model.

# We start with EDA, turn categorical explanatory variables to dummy variables, and consideer the application
# of PCA. Then we train each of Regression SVM, Random Forest and MLP, using re-sampling, and get three models, 
# trained with the respective best hyperparameters.
# In the end we compare the predictions of each best model using the validation set, and then select the best model to predict the test set.
# Along with the aformentioned processes, we show various plots explaining each step.

library(caret)
library(ggplot2)
library(gridExtra)
source("D:\\Statistica Applicata\\House Prices\\caret_analysis\\clean_dataset_house.R")

dataset = read.csv("D:\\Statistica Applicata\\House Prices\\train.csv")
summary(dataset)
head(dataset)

# -------------- Clean dataset -------------------------
sum(is.na(dataset))
dataset = clean_dataset_house(dataset)
sum(is.na(dataset))
summary(dataset)
dataset_expl = dataset[,(1:ncol(dataset))-1]
dataset_resp = dataset[,ncol(dataset)]

# -------------------------- PCA and Pre-processing ----------------------------------------------
# Now we proceed to apply PCA to the dataset, and split into train and test sets,
# The preprocess function will apply PCA only to the numerical variables
dataset_pca = preProcess(dataset_expl[,-1], method="pca", thresh=0.95)
dataset_expl = predict(dataset_pca, dataset_expl[,-1])
dataset = data.frame(dataset_expl, SalePrice = dataset_resp)
summary(dataset)
# --------------Split the dataset -------------------

set.seed(21111)
idx = createDataPartition(dataset$SalePrice, p=0.85, list = FALSE)
dataset_id = dataset[,"Id"]
trainset = dataset[idx,]
testset = dataset[-idx,]
testset_expl = testset[,(1:ncol(testset)-1)]
testset_resp = testset[,ncol(testset)]

# parallelism
library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

# ---------------------------- RANDOM FOREST MODEL ------------------------------------------------
# TRAIN AND TUNING
control = trainControl(method="cv", number=10)
grid = expand.grid(mtry=c(ncol(trainset)/3, sqrt(ncol(trainset))), splitrule="variance", min.node.size = c(5,10))
start_time <- Sys.time()
model.rf = train(SalePrice~., data=trainset, method="ranger", metric = "RMSE", tuneGrid = grid, trControl=control, importance="impurity")
# mtry=12 and mtry=54 were confronted using RMSE, computed on cross-validation with K=5, and mtry=54 is better.
end_time <- Sys.time()
end_time - start_time
plot(model.rf)


# VARIABLE IMPORTANCE
varImp(model.rf)

# PREDICTIONS
predictions = predict.train(model.rf, newdata = testset_expl)
pred_df = data.frame(Id = testset[,1], SalePrice = predictions)
write.csv(pred_df, "D:\\Statistica Applicata\\House Prices\\preds_rf.csv", row.names = FALSE, quote = FALSE)

# PERFORMANCE TEST
postResample(pred = predictions, obs = testset_resp)


# ------------------------------ Regression SVM ---------------------------------------
# For SVM, sigma is found automatically by an algorithm, while C is chosen by tuneLength. We train
control = trainControl(method="boot", number=25)
start_time <- Sys.time()
model.svm = train(SalePrice~., data=trainset, method="svmRadial", tuneLength=5, trControl = control, importance = TRUE)
end_time <- Sys.time()
end_time - start_time
plot(model.svm)
varImp(model.svm)

predictions = predict.train(model.svm, newdata = testset_expl)
pred_df = data.frame(Id = testset[,1], SalePrice = predictions)
write.csv(pred_df, "D:\\Statistica Applicata\\House Prices\\preds_svm.csv", row.names = FALSE, quote = FALSE)

postResample(predictions, testset_resp)

# Regression SVM scores worst than random forest
registerDoSEQ()
stopCluster(cl)
