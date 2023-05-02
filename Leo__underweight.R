# Load the required packages
library(e1071)
library(randomForest)
library(class)
library(ROCR)
library(zoo)
library(ROSE)
library(ggplot2)
library(caret)
library(e1071)
library(dplyr)
library(xts)
library(rpart)
library(ada)
library(adabag)
# Load the data
underwa <- read.csv("C:/Users/USER/Desktop/leonci samson/dfu_new.csv", header=TRUE)

# Remove any rows with missing values
underwa <- na.omit(underwa)

# Create a table of underweight counts
underweight_table <- table(underwa$underweight)

# Calculate the percentage of underweight and not underweight
underweight_pct <- round(prop.table(underweight_table) * 100)
underweight_labels <- paste0(c("Not underweight ", "Underweight "), underweight_pct, "%")

# Create a pie chart
pie(underweight_table, labels = underweight_labels,
    main = "Underweight Status", col = c("green", "red"))



########
#Fit model 1
#######



# Split the Data
set.seed(123)
trainIndex <- createDataPartition(underwa$underweight, p = 0.8, list = FALSE)
underwa_train <- underwa[trainIndex, ]
underwa_test <- underwa[-trainIndex, ]

# balancing
df_balanced <- ROSE(underweight ~ ., data = underwa_train, seed = 1)$data
####
table(df_balanced$underweight)
fred = table(df_balanced$underweight)
percent <- round(prop.table(fred) * 100)
# Create a pie chart of the frequencies
pie(fred, labels = paste(names(fred), percent, "%"), col = colors[as.factor(names(fred))], main = "Frequency of Underweight Variable for balanced datasets")
# Define colors for the pie chart
colors <- c("blue", "red")

##logistic regression

model <- glm(underweight~., data = df_balanced, family = binomial)

# Print the summary of the model
summary(model)
probabilities <- predict(model, newdata = underwa_test[,-1], type = "response")

# Predict the class based on a threshold of 0.5
predicted_class <- ifelse(probabilities > 0.5, 1, 0)


# Calculate accuracy of the model
actual_class <- underwa_test$underweight
accuracy <- mean(actual_class == predicted_class)

# Print the accuracy
accuracy

# Create a prediction object
predictions <- prediction(probabilities, actual_class)


# Create a performance object
performance <- performance(predictions, "tpr", "fpr")



# Plot the ROC curve
plot(performance, main = "ROC Curve",col = "blue")
abline(0, 1, col = "black") # Add the diagonal line

# Calculate the AUC
auc <- performance(predictions, "auc")@y.values[[1]]

# Add the legend
legend("bottomright", legend = paste("L R AUC =", round(auc, 2)), bty = "n", cex = 0.8, col = "black")

# Print the AUC
cat("AUC:", auc)

################################################3
# Load the required libraries


# Convert the factor levels of df_balanced$underweight to "0" and "1"
df_balanced$underweight <- factor(df_balanced$underweight, levels = c("0", "1"))

# Specify the training control
tr_control <- trainControl(method = "cv", number = 10)

# Train the KNN model using cross-validation
knn_model <- train(underweight ~ ., data = df_balanced, method = "knn", trControl = tr_control)

# Make probability predictions on the training set using cross-validation
knn_prob <- predict(knn_model, newdata = df_balanced, type = "prob")

# Convert the probability predictions to class predictions
knn_pred <- ifelse(knn_prob[, "1"] > 0.5, "1", "0")

# Convert the class predictions and the reference data to factors with the same levels
knn_pred <- factor(knn_pred, levels = c("0", "1"))
df_balanced$underweight <- factor(df_balanced$underweight, levels = c("0", "1"))

# Evaluate the model using different metrics
confusionMatrix(knn_pred, df_balanced$underweight)
precision <- posPredValue(knn_pred, df_balanced$underweight, positive = "1")
recall <- sensitivity(knn_pred, df_balanced$underweight, positive = "1")
f1_score <- (2 * precision * recall) / (precision + recall)
auc_roc <- performance(prediction(knn_prob[, "1"], df_balanced$underweight), "auc")@y.values[[1]]

# Print the evaluation metrics
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-score:", f1_score, "\n")
cat("AUC-ROC:", auc_roc, "\n")


library(ROCR)

# Create a prediction object
knn_roc <- prediction(knn_prob[, "1"], df_balanced$underweight)

# Create a performance object
knn_perf <- performance(knn_roc, "tpr", "fpr")

# Plot the ROC curve
plot(knn_perf, main = "ROC Curve for KNN Model", col = "blue", lwd = 2)

# Add the AUC-ROC to the plot
auc_knn <- as.numeric(performance(knn_roc, "auc")@y.values)
# Create a prediction object
knn_roc <- prediction(knn_prob[, "1"], df_balanced$underweight)

# Create a performance object
knn_perf <- performance(knn_roc, "tpr", "fpr")

# Plot the ROC curve
plot(knn_perf, main = "ROC Curve for KNN Model", col = "blue", lwd = 2)

#################################################################

# Train the random forest model
rf_model <- randomForest(underweight ~ ., data = df_balanced, ntree = 500, importance = TRUE)

# Make predictions on the testing set
rf_pred <- predict(rf_model, newdata = underwa_test)
##
# Check the data type of the target variable in the testing set
class(underwa_test$underweight)

# Convert the target variable in the testing set to a factor with appropriate levels
underwa_test$underweight <- factor(as.character(underwa_test$underweight), levels = c("0", "1"))
# Evaluate the model
confusionMatrix(rf_pred, underwa_test$underweight)

# Calculate the importance of each predictor
varImpPlot(rf_model)
# Make probability predictions on the testing set
rf_prob <- predict(rf_model, newdata = underwa_test, type = "prob")

# Create the prediction object
rf_pred_obj <- prediction(rf_prob[, 2], underwa_test$underweight)

# Create the performance object
rf_perf <- performance(rf_pred_obj, "tpr", "fpr")

# Plot the ROC curve
plot(rf_perf, main = "ROC Curve for Random Forest Classifier", col = "blue")
abline(0, 1, col = "black") # Add the diagonal line

# Calculate the AUC
rf_auc <- performance(rf_pred_obj, "auc")@y.values[[1]]
cat("AUC:", rf_auc, "\n")

# Add the legend
legend("bottomright", legend = paste("RF AUC =", round(rf_auc, 2)),  bty = "n", cex = 0.8, col = "black")

##############################################################################
# Load the caret package
library(caret)
library(e1071)
# Fit an SVM model on the training set
svm_model <- svm(underweight ~ ., data = df_balanced)

# Make predictions on the testing set
svm_pred <- predict(svm_model, newdata = underwa_test)

# Compute the confusion matrix
conf_mat <- confusionMatrix(svm_pred, underwa_test$underweight)

# Extract the accuracy, sensitivity, specificity, precision, recall, and F1 score
acc <- conf_mat$overall["Accuracy"]
sens <- conf_mat$byClass["Sensitivity"]
spec <- conf_mat$byClass["Specificity"]
prec <- conf_mat$byClass["Precision"]
rec <- conf_mat$byClass["Recall"]
f1 <- conf_mat$byClass["F1"]

# Print the performance metrics
cat("Accuracy:", acc, "\n")
cat("Sensitivity:", sens, "\n")
cat("Specificity:", spec, "\n")
cat("Precision:", prec, "\n")
cat("Recall:", rec, "\n")
cat("F1 Score:", f1, "\n")
######To plot the ROC curves for the SVM, Random Forest, and KNN classifiers on the same plot and include a legend with accuracy and AUC values, you can use the following code:
# Fit an SVM model on the training set with probability estimates
library(ROCR)
# Fit an SVM model on the training set
svm_model <- svm(underweight ~ ., data = df_balanced, probability = TRUE)
# Compute the ROC curve and AUC
svm_pred_prob <- attr(predict(svm_model, newdata = underwa_test, probability = TRUE), "probabilities")[, 2]
svm_roc <- prediction(svm_pred_prob, underwa_test$underweight)
svm_auc <- performance(svm_roc, "auc")@y.values[[1]]

# Plot ROC curve
svm_perf <- performance(svm_roc, "tpr", "fpr")
plot(svm_perf, col = "blue", lwd = 2, main = "ROC of SVM Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")
legend("bottomright", legend = paste("AUC =", round(svm_auc, 2)), col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2)

####################################################
####################################################
# Load the ROCR package
library(ROCR)

# Fit the Ada model
ada_model <- ada(underweight ~ ., data = df_balanced)


# Make predictions on the test set
ada_pred_prob <- predict(ada_model, newdata = underwa_test, type = "prob")[,2]

# Make predictions on the test set
ada_pred <- predict(ada_model, newdata = underwa_test)

# Create confusion matrix
cm <- table(ada_pred, underwa_test$underweight)
colnames(cm) <- c("Not Underweight", "Underweight")
rownames(cm) <- c("Not Underweight", "Underweight")
print(cm)

# Calculate accuracy
accuracy <- sum(diag(cm)) / sum(cm)
print(paste0("Accuracy: ", round(accuracy, 3)))

# Calculate sensitivity (recall)
sensitivity <- cm[2, 2] / sum(cm[2, ])
print(paste0("Sensitivity: ", round(sensitivity, 3)))

# Calculate specificity
specificity <- cm[1, 1] / sum(cm[1, ])
print(paste0("Specificity: ", round(specificity, 3)))

# Calculate precision
precision <- cm[2, 2] / sum(cm[, 2])
print(paste0("Precision: ", round(precision, 3)))

# Calculate F1 score
f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
print(paste0("F1 Score: ", round(f1_score, 3)))


# Compute the ROC curve and AUC
roc_obj <- performance(prediction(ada_pred_prob, underwa_test$underweight), "tpr", "fpr")
auc_val <- round(as.numeric(performance(prediction(ada_pred_prob, underwa_test$underweight), "auc")@y.values), 3)

# Plot the ROC curve and add legend
# Plot the ROC curve and add legend
plot(roc_obj, main = "ROC Curve - Ada Model", print.cutofat = seq(0, 1, by = 0.1), print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, col = "blue")

legend("bottomright", legend = paste("ADA AUC =", auc_val), bty = "n")
abline(a = 0, b = 1, lty = 2, col = "gray")

####################################################

library(ROCR)
par(mfrow=c(2,2))

# Plot for Logistic Regression Model
plot(performance, main = "ROC Curve - Logistic Regression Model",col = "blue")
abline(0, 1, col = "black")
auc <- performance(predictions, "auc")@y.values[[1]]
legend("bottomright", legend = paste("L R AUC =", round(auc, 2)), bty = "n", cex = 0.8, col = "black")

# Plot for KNN Model
plot(knn_perf, main = "ROC Curve - KNN Model", col = "blue", lwd = 2)
auc_knn <- as.numeric(performance(knn_roc, "auc")@y.values)
legend("bottomright", legend = paste("KNN AUC =", round(auc_knn, 2)),  bty = "n", cex = 0.8, col = "black")
abline(a = 0, b = 1, lty = 2)
# Plot for Random Forest Model
plot(rf_perf, main = "ROC Curve - Random Forest Model", col = "blue")
abline(0, 1, col = "black")
rf_auc <- performance(rf_pred_obj, "auc")@y.values[[1]]
legend("bottomright", legend = paste("RF AUC =", round(rf_auc, 2)),  bty = "n", cex = 0.8, col = "black")

# Plot for SVM Model
plot(svm_perf, col = "blue", lwd = 2, main = "ROC Curve - SVM Model", xlab = "False Positive Rate", ylab = "True Positive Rate")
svm_auc <- performance(svm_roc, "auc")@y.values[[1]]
legend("bottomright", legend = paste("SVM AUC =", round(svm_auc, 2)), col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2)
# Plot the ROC curve and add legend of ADA model
plot(roc_obj, main = "ROC Curve - Ada Model", print.cutofat = seq(0, 1, by = 0.1), print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, col = "blue")

legend("bottomright", legend = paste("ADA AUC =", auc_val), bty = "n")
abline(a = 0, b = 1, lty = 2, col = "gray")
###########################################################
############################################# ############
# Combine the ROC curves into one graph
plot(performance, col = "blue", lty = 1, lwd = 2,
     main = "ROC Curves for Different Models")
lines(knn_perf, col = "red", lty = 2, lwd = 2)
lines(rf_perf, col = "green", lty = 3, lwd = 2)
lines(svm_perf, col = "purple", lty = 4, lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "KNN", "Random Forest", "SVM"), 
       col = c("blue", "red", "green", "purple"), lty = c(1, 2, 3, 4), 
       cex = 0.8, bty = "n")
