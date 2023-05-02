library(class)
library(MASS)
library(lattice)
library(ggplot2)
library(kernlab)
library(mlbench)
library(reshape2)
library(ROCR)
library(rpart)
library(ada)
library(adabag)
library(caret)
library(ipred)
library(survival)
library(rchallenge)
library(PerformanceAnalytics)
library(knitr)
library(acepack)
library(HSAUR2)
library(corrplot)
library(entropy)
#library(binda)
library(gbm)
library(xgboost)
library(randomForest)
# Load your data into a data frame
data <- read.csv("C:/Users/USER/Desktop/leonci samson/dfs_new.csv", header = TRUE)
# Remove rows with missing values
data <- na.omit(data)

# Create a table of stunting counts
stunting_table <- table(data$stunting)

# Calculate the percentage of stunted and not stunted
stunting_pct <- round(prop.table(stunting_table) * 100, 1)

# Create a pie chart
pie(stunting_table, labels = paste0(c("Not Stunted ", "Stunted "), stunting_pct, "%"),
    main = "Stunting Status", col = c("green", "red"))

# Add a legend
legend("topright", legend = c("Not Stunted", "Stunted"), 
       cex = 0.8, fill = c("green", "red"))


########
#Fit model 1
#######
# Load the caret package
library(caret)

# Set the seed for reproducibility
set.seed(123)

# Split the data into 80% for training and 20% for testing
trainIndex <- createDataPartition(data$stunting, p = 0.8, list = FALSE, times = 1)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]


# create a logistic regression model
logreg <- glm(stunting ~ ., data = train, family = binomial())

# predict on the test data
y_pred <- predict(logreg, newdata = train, type = "response")

y_predMode1 <- ifelse(y_pred > 0.5,1,0)

########
#Fit model 2
########

# convert stunting to a factor
train$stunting <- as.factor(train$stunting)
test$stunting <- as.factor(test$stunting)
# Define the training control
train_control <- trainControl(method = "cv", number = 5)

# fit random forest model
rpart_model <- train(stunting ~ ., data=train, method="rpart", trControl=trainControl(method = "cv", number = 5), tuneLength=5)

# Make predictions on the test set
y_pred <- predictions <- predict(rpart_model, newdata=train)

y_predMode2 <- y_pred

#####
#Fit model 3
#####

# fit random forest model
rf_model <- train(stunting ~ ., data=train, method="rf", trControl=trainControl(method = "cv", number = 5), tuneLength=5)

# Make predictions on the test set
y_pred <- predictions <- predict(rf_model, newdata=train)

y_predMode3 <- y_pred



#####
#Model 4
####


# fit random forest model
ada_model <- train(stunting ~ ., data=train, method="AdaBoost.M1", trControl=trainControl(method = "cv", number = 5), tuneLength=5)

# Make predictions on the test set
y_predMode4 <- predictions <- predict(ada_model, newdata=train)
########################################################################
#model 5
##############################################################################
# fit random forest model
svm_model <- train(stunting ~ ., data = train, method = "svmRadial", trControl = train_control)
# Make predictions on the test set
y_predMode5 <- predictions <- predict(svm_model, newdata = train)

##################################
#model 6
#######################################
# fit random forest model
knn_model <- train(stunting ~ ., data = train, method = "knn", trControl = train_control)
# Make predictions on the test set
y_predMode6 <- predictions <- predict(knn_model, newdata = train)

#################################################
#model7
######################################################
#fit random forest model
nnet_model <- train(stunting ~ ., data = train, method = "nnet", trControl = train_control)
# Make predictions on the test set
y_predMode7 <- predictions <- predict(nnet_model, newdata = train)

############################################################
#model8
#####################################################
gbm_model <- train(stunting ~ ., data = train, method = "gbm", trControl = train_control)
# Make predictions on the test set
y_predMode8 <- predictions <- predict(gbm_model, newdata = train)
##########################################################
#model9
###########################################################
xgb_model <- train(stunting ~ ., data = train, method = "xgbTree", trControl = train_control)
# Make predictions on the test set
y_predMode9 <- predictions <- predict(xgb_model, newdata = train)

#train.output = data.frame(stunting = train$stunting,model1 =y_predMode4,
# model2 =y_predMode2,
#model3 = y_predMode3,
#model4 = y_predMode5)
train.output <- data.frame(
  stunting = test$stunting,
  model1 = y_predMode4[1:nrow(test)],
  model2 = y_predMode2[1:nrow(test)],
  model3 = y_predMode3[1:nrow(test)],
  model4 = y_predMode5[1:nrow(test)],
  model5 = y_predMode6[1:nrow(test)],
  model6 = y_predMode7[1:nrow(test)],
  model7 = y_predMode8[1:nrow(test)],
  model8 = y_predMode9[1:nrow(test)]
)




### Combine model

Final.tree = train(stunting ~ ., 
                   data=train.output, 
                   method="rpart", 
                   trControl = trainControl(method = "cv"))


##############
#Test model

y_pred <- predict(logreg, newdata = test, type = "response")
y_predMode1 <- ifelse(y_pred > 0.5,1,0)

y_predMode2 <- predict(rpart_model, newdata=test)
y_predMode3 <- predictions <- predict(rf_model, newdata=test)
y_predMode4 <- predictions <- predict(ada_model, newdata=test)
y_predMode5 <- predictions <- predict(svm_model, newdata = test)
y_predMode6 <- predictions <- predict(knn_model, newdata = test)
y_predMode7 <- predictions <- predict(nnet_model, newdata = test)
y_predMode8 <- predictions <- predict(gbm_model, newdata = test)
y_predMode9 <- predictions <- predict(xgb_model, newdata = test)
##############

test.output = data.frame(model1  =y_predMode2,
                         model2 =y_predMode3,
                         model3  = y_predMode4,
                         model4 = y_predMode5,
                         model5 = y_predMode6,
                         model6 = y_predMode7,
                         model7 = y_predMode8,
                         model8 = y_predMode9
)

y_pred <- predictions <- predict(Final.tree, newdata=test.output)


# Create a confusion matrix
confusionMatrix(table(test$stunting, y_pred))

#######################################
##########################################
##########################################
# Load the necessary libraries
library(caret)
library(mlbench)



# Fit model 1: Logistic Regression
varImp(logreg)

# Fit model 2: Random Forest
varImp(rf_model)

# Fit model 3: Support Vector Machine
varImp(svm_model)

# Fit model 4: Gradient Boosting Machine
varImp(gbm_model)

# Fit model 5: XGBoost
varImp(xgb_model)

# Fit model 6: k-Nearest Neighbors

varImp(knn_model)

# Fit model 7: AdaBoost

varImp(ada_model)

# Fit model 8: Neural Network

varImp(nnet_model)

# Fit model 9: Classification Tree

varImp(rpart_model)

# Create a matrix of feature importance
importance_matrix <- matrix(0, nrow = 9, ncol = 9)
rownames(importance_matrix) <- c("Logistic Regression", "Random Forest", "Support Vector Machine", "Gradient Boosting Machine", "XGBoost", "k-Nearest Neighbors", "AdaBoost", "Neural Network", "Classification Tree")
colnames(importance_matrix) <- colnames(train[, -ncol(train)])
importance_matrix[1, ] <- varImp(logreg)$importance[, 1]
importance_matrix[2, ] <- varImp(rf_model)$importance[, 1]
importance_matrix[3, ] <- varImp(svm_model)$importance[, 1]
importance_matrix[4, ] <- varImp(gbm_model)$importance[, 1]
importance_matrix[5, ] <- varImp(xgb_model)$importance[, 1]
importance_matrix[6, ] <- varImp(knn_model)$importance[, 1]
importance_matrix[7, ] <- varImp(ada_model)$importance[, 1]
importance_matrix[8, ] <- varImp(nnet_model)$importance[, 1]
importance_matrix[9, ] <- varImp(rpart_model)$importance[, 1]
importance_matrix
