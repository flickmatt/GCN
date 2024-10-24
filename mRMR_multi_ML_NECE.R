# mRMRe and multiple ML algs
# WITHOUT TUNING ----
# Required Libraries
library(mRMRe)
library(caret)
library(pROC)
library(randomForest)
library(e1071)
library(xgboost)
library(ggplot2)

# Step 1: Select columns that contain "Raw_Mean" but not "nii.gz" or "SPGRC"
df_cols <- grep("Raw_Mean", names(master), value = TRUE)
df_cols <- df_cols[!grepl("nii.gz", df_cols)]
df_cols <- df_cols[!grepl("SPGRC", df_cols)]
df_cols <- c(df_cols, "MRI.contrast.enhancing.annotation", "Iavarone_ID")

# Subset the dataframe and scale the features
df <- na.omit(master[, df_cols])
cols_to_scale <- setdiff(df_cols, c("MRI.contrast.enhancing.annotation", "Iavarone_ID"))
df[, cols_to_scale] <- scale(df[, cols_to_scale])

# Step 2: Split data into training and test sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(df$MRI.contrast.enhancing.annotation, p = 0.7, list = FALSE)
train_data_samples <- df[train_index, ]
train_data <- data.frame(train_data_samples[1:53])  # Limit to first 53 features
test_data_samples <- df[-train_index, ]
test_data <- data.frame(test_data_samples[1:53])

# Set the target index
target_index <- ncol(train_data)

# Convert the target column to an ordered factor
train_data[, target_index] <- as.ordered(train_data[, target_index])

# Convert all feature columns (except the target) to numeric where possible
train_data[-target_index] <- lapply(train_data[-target_index], function(col) {
  if (is.character(col)) {
    as.numeric(as.character(col))  # Convert characters to numeric
  } else if (is.factor(col)) {
    as.ordered(col)  # Convert unordered factors to ordered factors
  } else {
    col
  }
})

# Step 3: mRMR feature selection
mrmr_data <- mRMR.data(data = train_data)
mrmr_result <- mRMR.classic(data = mrmr_data, target_indices = target_index, feature_count = 53)

# Filter features with positive mRMR scores
feature_scores <- mrmr_result@scores[[1]]
selected_features <- mrmr_result@filters[[1]]
positive_score_indices <- selected_features[feature_scores > 0]
ranked_features <- colnames(df)[positive_score_indices]
ranked_features

# Step 4: Cross-Validation Setup
control <- trainControl(method = "cv", 
                        number = 5, 
                        classProbs = TRUE,  # To get probabilities for ROC curve
                        summaryFunction = twoClassSummary, 
                        savePredictions = TRUE)  # Save predictions to create ROC curve later

# Step 5: Apply Multiple Algorithms

# (1) Random Forest
rf_model <- train(as.formula(paste("MRI.contrast.enhancing.annotation", "~ .")),
                  data = train_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")],
                  method = "rf",
                  metric = "ROC",
                  trControl = control)

# (2) Support Vector Machine (SVM)
svm_model <- train(as.formula(paste("MRI.contrast.enhancing.annotation", "~ .")),
                   data = train_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")],
                   method = "svmRadial",
                   metric = "ROC",
                   trControl = control)

# (3) Gradient Boosting (XGBoost)
xgb_model <- train(as.formula(paste("MRI.contrast.enhancing.annotation", "~ .")),
                   data = train_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")],
                   method = "xgbTree",
                   metric = "ROC",
                   trControl = control)

# Step 6: Extract predictions from cross-validation and plot ROC for each model

# Random Forest ROC
rf_predictions <- rf_model$pred
rf_roc <- roc(rf_predictions$obs, rf_predictions$CE)
plot(rf_roc, legacy.axes = T, main = "Cross Validation ROC of Random Forest", print.auc = T)

# SVM ROC
svm_predictions <- svm_model$pred
svm_roc <- roc(svm_predictions$obs, svm_predictions$CE)
plot(svm_roc, legacy.axes = T, main = "Cross Validation ROC of SVM", print.auc = T)

# XGBoost ROC
xgb_predictions <- xgb_model$pred
xgb_roc <- roc(xgb_predictions$obs, xgb_predictions$CE)
plot(xgb_roc, legacy.axes = T, main = "Cross Validation ROC of XGBoost", print.auc = T)

# Step 7: Test Set ROC for Random Forest
rf_test_probabilities <- predict(rf_model, newdata = test_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")], type = "prob")
rf_test_roc <- roc(test_data$MRI.contrast.enhancing.annotation, rf_test_probabilities$CE)
plot(rf_test_roc, legacy.axes = T, main = "Test Set ROC of Random Forest", print.auc = T)

# Step 8: Test Set ROC for SVM
svm_test_probabilities <- predict(svm_model, newdata = test_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")], type = "prob")
svm_test_roc <- roc(test_data$MRI.contrast.enhancing.annotation, svm_test_probabilities$CE)
plot(svm_test_roc, legacy.axes = T, main = "Test Set ROC of SVM", print.auc = T)

# Step 9: Test Set ROC for XGBoost
xgb_test_probabilities <- predict(xgb_model, newdata = test_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")], type = "prob")
xgb_test_roc <- roc(test_data$MRI.contrast.enhancing.annotation, xgb_test_probabilities$CE)
plot(xgb_test_roc, legacy.axes = T, main = "Test Set ROC of XGBoost", print.auc = T)

# 

# WITH TUNING ----
# Load required libraries
library(mRMRe)
library(caret)
library(glmnet)  # For logistic regression
library(pROC)
library(ggplot2)

# Select columns that contain "Raw_Mean" but not "nii.gz"
df_cols <- grep("Raw_Mean", names(master), value = TRUE)
df_cols <- df_cols[!grepl("nii.gz", df_cols)]
df_cols <- df_cols[!grepl("SPGRC", df_cols)]

# Add the 'MRI.contrast.enhancing.annotation' column to the selection
df_cols <- c(df_cols, "MRI.contrast.enhancing.annotation", "Iavarone_ID")

# Subset the dataframe and clean data
df <- na.omit(master[, df_cols])

# Identify columns to scale
cols_to_scale <- setdiff(df_cols, c("MRI.contrast.enhancing.annotation", "Iavarone_ID"))
df[, cols_to_scale] <- scale(df[, cols_to_scale])

# Split into training and testing sets
set.seed(123)
train_index <- createDataPartition(df$MRI.contrast.enhancing.annotation, p = 0.7, list = FALSE)
train_data_samples <- df[train_index, ]
train_data <- data.frame(train_data_samples[1:53])  # Limit to first 53 features
test_data_samples <- df[-train_index, ]
test_data <- data.frame(test_data_samples[1:53])

# Set target index
target_index <- ncol(train_data)
train_data[, target_index] <- as.ordered(train_data[, target_index])

# Convert all feature columns (except target) to numeric where possible
train_data[-target_index] <- lapply(train_data[-target_index], function(col) {
  if (is.character(col)) {
    as.numeric(as.character(col))  # Convert characters to numeric
  } else if (is.factor(col)) {
    as.ordered(col)  # Convert unordered factors to ordered factors
  } else {
    col
  }
})

# mRMR feature selection
mrmr_data <- mRMR.data(data = train_data)
mrmr_result <- mRMR.classic(data = mrmr_data, target_indices = target_index, feature_count = 53)
feature_scores <- mrmr_result@scores[[1]]
selected_features <- mrmr_result@filters[[1]]
positive_score_indices <- selected_features[feature_scores > 0]
ranked_features <- colnames(df)[positive_score_indices]

# Cross-validation setup with hyperparameter tuning
control <- trainControl(method = "cv", 
                        number = 5, 
                        classProbs = TRUE,  
                        summaryFunction = twoClassSummary, 
                        savePredictions = TRUE,
                        search = "grid")  # Use grid search for tuning

# 1. Logistic Regression Tuning
logistic_grid <- expand.grid(alpha = c(0, 0.5, 1),  # alpha = 0 (Ridge), 1 (Lasso), 0.5 (Elastic Net)
                             lambda = c(0.001, 0.01, 0.1, 1, 10))  # Regularization strength

logistic_model <- train(as.formula(paste("MRI.contrast.enhancing.annotation", "~ .")),
                        data = train_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")],
                        method = "glmnet",
                        metric = "ROC",
                        tuneGrid = logistic_grid,  # Grid of hyperparameters
                        trControl = control)

# 2. Random Forest Tuning
rf_grid <- expand.grid(mtry = c(1, 3, 5, 7, 9))  # Example grid of 'mtry' values for tuning

rf_model <- train(as.formula(paste("MRI.contrast.enhancing.annotation", "~ .")),
                  data = train_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")],
                  method = "rf",
                  metric = "ROC",
                  tuneGrid = rf_grid,  # Grid of hyperparameters
                  trControl = control)

# 3. SVM Radial Tuning
svm_grid <- expand.grid(sigma = c(0.01, 0.1, 1), C = c(1, 10, 100))  # Grid for radial SVM

svm_model <- train(as.formula(paste("MRI.contrast.enhancing.annotation", "~ .")),
                   data = train_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")],
                   method = "svmRadial",
                   metric = "ROC",
                   tuneGrid = svm_grid,  # Grid of hyperparameters
                   trControl = control)

# 4. XGBoost Tuning
xgb_grid <- expand.grid(nrounds = c(50, 100, 150),
                        max_depth = c(2, 4, 6),
                        eta = c(0.01, 0.1, 0.3),
                        gamma = c(0, 1, 5),
                        colsample_bytree = c(0.5, 0.7),
                        min_child_weight = c(1, 3),
                        subsample = c(0.6, 0.8))

xgb_model <- train(as.formula(paste("MRI.contrast.enhancing.annotation", "~ .")),
                   data = train_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")],
                   method = "xgbTree",
                   metric = "ROC",
                   tuneGrid = xgb_grid,  # Grid of hyperparameters
                   trControl = control)

# Evaluation: Compute ROC for cross-validated predictions and test set for each model
library(pROC)

# Function to compute and plot ROC for cross-validation
plot_roc <- function(model, test_data, ranked_features) {
  cv_predictions <- model$pred
  roc_curve <- roc(cv_predictions$obs, cv_predictions$CE)  # Replace 'CE' with your positive class
  plot(roc_curve, legacy.axes = TRUE, print.auc = TRUE, main = paste("Cross Validation ROC -", model$method))
  
  # Test set ROC
  test_probabilities <- predict(model, newdata = test_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")], type = "prob")
  test_roc_curve <- roc(test_data$MRI.contrast.enhancing.annotation, test_probabilities$CE)
  plot(test_roc_curve, legacy.axes = TRUE, print.auc = TRUE, main = paste("Test Set ROC -", model$method))
}

# Plot ROC for each model
plot_roc(logistic_model, test_data, ranked_features)
plot_roc(rf_model, test_data, ranked_features)
plot_roc(svm_model, test_data, ranked_features)
plot_roc(xgb_model, test_data, ranked_features)

