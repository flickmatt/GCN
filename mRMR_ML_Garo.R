# mRMR_Garo_ML_GCN
# SPLIT, MRMR, ML (NEURONAL) ----
master <- read.table("/Users/m254284/Desktop/for_matt/master.tsv", sep = "\t", header = T)
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
df_cols <- df_cols[!grepl("QWS", df_cols)]
df_cols <- df_cols[!grepl("UCLA", df_cols)]
df_cols <- c(df_cols, "GarofanoClass.y", "Iavarone_ID")

# Subset the dataframe and scale the features
df <- na.omit(master[, df_cols])
df <- df[df$GarofanoClass.y != "nc", ]
df$GarofanoClass.y <- ifelse(df$GarofanoClass.y == "Neuronal", "Neuronal", "Other")
cols_to_scale <- setdiff(df_cols, c("Iavarone_ID", "GarofanoClass.y"))
df[, cols_to_scale] <- scale(df[, cols_to_scale])

# Step 2: Split data into training and test sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(df$GarofanoClass.y, p = 0.7, list = FALSE)
train_data_samples <- df[train_index, ]
train_data <- data.frame(train_data_samples[1:34])  # Limit to first 33 features
test_data_samples <- df[-train_index, ]
test_data <- data.frame(test_data_samples[1:34])

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
mrmr_result <- mRMR.classic(data = mrmr_data, target_indices = target_index, feature_count = 33)

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

# Step 5: Multiple Algorithms Applied

# 1. Logistic Regression Tuning
logistic_grid <- expand.grid(alpha = c(0, 0.5, 1),  # alpha = 0 (Ridge), 1 (Lasso), 0.5 (Elastic Net)
                             lambda = c(0.001, 0.01, 0.1, 1, 10))  # Regularization strength

logistic_model <- train(as.formula(paste("GarofanoClass.y", "~ .")),
                        data = train_data[, c(ranked_features, "GarofanoClass.y")],
                        method = "glmnet",
                        metric = "ROC",
                        tuneGrid = logistic_grid,  # Grid of hyperparameters
                        trControl = control)

# 2. Random Forest Tuning
rf_grid <- expand.grid(mtry = c(1, 3, 5, 7, 9))  # Example grid of 'mtry' values for tuning

rf_model <- train(as.formula(paste("GarofanoClass.y", "~ .")),
                  data = train_data[, c(ranked_features, "GarofanoClass.y")],
                  method = "rf",
                  metric = "ROC",
                  tuneGrid = rf_grid,  # Grid of hyperparameters
                  trControl = control)

# 3. SVM Radial Tuning
svm_grid <- expand.grid(sigma = c(0.01, 0.1, 1), C = c(1, 10, 100))  # Grid for radial SVM

svm_model <- train(as.formula(paste("GarofanoClass.y", "~ .")),
                   data = train_data[, c(ranked_features, "GarofanoClass.y")],
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

xgb_model <- train(as.formula(paste("GarofanoClass.y", "~ .")),
                   data = train_data[, c(ranked_features, "GarofanoClass.y")],
                   method = "xgbTree",
                   metric = "ROC",
                   tuneGrid = xgb_grid,  # Grid of hyperparameters
                   trControl = control)

# Evaluation: Compute ROC for cross-validated predictions and test set for each model
library(pROC)

# Function to compute and plot ROC for cross-validation
plot_roc <- function(model, test_data, ranked_features) {
  cv_predictions <- model$pred
  roc_curve <- roc(cv_predictions$obs, cv_predictions$Neuronal)
  plot(roc_curve, legacy.axes = TRUE, print.auc = TRUE, main = paste("Cross Validation ROC -", model$method))
  
  # Test set ROC
  test_probabilities <- predict(model, newdata = test_data[, c(ranked_features, "GarofanoClass.y")], type = "prob")
  test_roc_curve <- roc(test_data$GarofanoClass.y, test_probabilities$Neuronal)
  plot(test_roc_curve, legacy.axes = TRUE, print.auc = TRUE, main = paste("Test Set ROC -", model$method))
}

# Plot ROC for each model
plot_roc(logistic_model, test_data, ranked_features)
plot_roc(rf_model, test_data, ranked_features)
plot_roc(svm_model, test_data, ranked_features)
plot_roc(xgb_model, test_data, ranked_features)

# combined plot
# Function to compute ROC for cross-validation
compute_roc <- function(model, test_data, ranked_features) {
  # Cross-Validation ROC
  cv_predictions <- model$pred
  roc_curve_cv <- roc(cv_predictions$obs, cv_predictions$Neuronal)  # Replace 'CE' with positive class
  
  # Test set ROC
  test_probabilities <- predict(model, newdata = test_data[, c(ranked_features, "GarofanoClass.y")], type = "prob")
  roc_curve_test <- roc(test_data$GarofanoClass.y, test_probabilities$Neuronal)
  
  return(list(cv = roc_curve_cv, test = roc_curve_test))
}

# Compute ROC curves for all models
roc_logistic <- compute_roc(logistic_model, test_data, ranked_features)
roc_rf <- compute_roc(rf_model, test_data, ranked_features)
roc_svm <- compute_roc(svm_model, test_data, ranked_features)
roc_xgb <- compute_roc(xgb_model, test_data, ranked_features)

# Plot all ROC curves on one plot, adjusting AUC positions to avoid overlap
plot(roc_logistic$test, col = "blue", legacy.axes = TRUE, main = "Test Set ROC Comparison (Neuronal)", 
     print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.55, lwd = 2)  # Logistic Regression

plot(roc_rf$test, add = TRUE, col = "red", 
     print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.5, lwd = 2)  # Random Forest

plot(roc_svm$test, add = TRUE, col = "green", 
     print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.45, lwd = 2)  # SVM

plot(roc_xgb$test, add = TRUE, col = "purple", 
     print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.4, lwd = 2)  # XGBoost

# Add a legend to the plot
legend("bottomright", legend = c("Logistic Regression", "Random Forest", "SVM", "XGBoost"),
       col = c("blue", "red", "green", "purple"), lwd = 2)


# EVENT GENERATION FOR DATA (NEURONAL) ----
library(tidyverse)
library(tibble)
library(gsubfn)
library(dendextend)

get_img_events <- function(img_df) {

  # Dissimilarity matrix based on euclidean distance
  d <- dist(img_df, method = "euclidean")
  # Hierarchical clustering using median linkage
  hc1 <- hclust(d, method = "median" )
  # expression rate dendrogram
  dend <- as.dendrogram(hc1)
  # Return all the nodes in the tree with corresponding samples
  events <- partition_leaves(dend)
  
  return(events)
}
#all data into neo4j
data_neo4j <- rbind(train_data_samples, test_data_samples)
data_neo4j <- data_neo4j[, c(ranked_features, "GarofanoClass.y", "Iavarone_ID")]
data_neo4j_optimal <- data_neo4j[, c(ranked_features, "GarofanoClass.y", "Iavarone_ID")]

img_names <- setdiff(colnames(data_neo4j), c("GarofanoClass.y", "Iavarone_ID"))
img_names_optimal <- setdiff(colnames(data_neo4j_optimal), c("GarofanoClass.y", "Iavarone_ID"))

# ALL SAMPLES W ALL FEATURES
setwd("/Users/m254284/Desktop/Neo4j_Projects/Img_events/ML/GCN_GARO/Neuronal")

for (img_no in img_names) {
  # Print the img name and the number
  print(paste("Working on img no.", img_no))
  
  # Get the img data
  img_df <- as.data.frame(data_neo4j[img_no])
  rownames(img_df) <- data_neo4j$Iavarone_ID
  img_name <- colnames(img_df)
  
  # Get the img events
  img_events <- get_img_events(img_df = img_df)
  
  # Create an empty data frame to append the result of each img tree event
  df <- data.frame(matrix(ncol = 11, nrow = 0))  # Adjusted number of columns
  
  for (i in seq(1, length(img_events))) {
    # Get event characteristics
    event <- i
    
    # Get event patients (samples)
    unlisted_patients <- unlist(img_events[i])
    patients <- paste(unlisted_patients, collapse=",")
    
    # Get median img signal for the event
    patients_ge <- img_df %>% filter(row.names(img_df) %in% unlisted_patients)
    median_ge <- median(patients_ge[[1]])
    
    # Get the number of patients and if there is only one patient, mark as a leaf
    no_of_patients <- length(unlisted_patients)
    leaf_status <- ifelse(no_of_patients == 1, 1, 0)
    
    # Determine subtype
    tissue_annotations <- data_neo4j %>%
      filter(Iavarone_ID %in% unlisted_patients) %>%
      select(GarofanoClass.y)
    
    # Calculate the percentage of NE and CE samples
    percentage_neuronal <- sum(tissue_annotations$GarofanoClass.y == "Neuronal") / no_of_patients * 100
    percentage_other <- sum(tissue_annotations$GarofanoClass.y == "Other") / no_of_patients * 100
    
    # Tissue region can be recorded as "All" if we want to generalize
    tissue_region <- "All"
    
    # Append the event data to the dataframe
    df <- rbind(df, c(event, patients, median_ge, no_of_patients, img_name, 'img_value', leaf_status, tissue_region, percentage_neuronal, percentage_other))
  }
  
  # Provide column names (adjusted for the new columns)
  colnames(df) <- c('event', 'samples', 'median_signal', 'no_of_patients', 'img_name', 'molecular_data', 'leaf_status', 'tissue_region', 'percentage_neuronal', 'percentage_other')
  
  # Save each img's event data to a CSV file
  file_name <- paste(img_name, ".csv", sep="")
  write.csv(df, file=file_name, row.names = FALSE)
}

# SPLIT, MRMR, ML (GLYCOLYTIC) ----
# master <- read.table("/Users/m254284/Desktop/for_matt/master.tsv", sep = "\t", header = T)
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
df_cols <- df_cols[!grepl("QWS", df_cols)]
df_cols <- df_cols[!grepl("UCLA", df_cols)]
df_cols <- c(df_cols, "GarofanoClass.y", "Iavarone_ID")

# Subset the dataframe and scale the features
df <- na.omit(master[, df_cols])
df <- df[df$GarofanoClass.y != "nc", ]
df$GarofanoClass.y <- ifelse(df$GarofanoClass.y == "Glycolytic", "Glycolytic", "Other")
cols_to_scale <- setdiff(df_cols, c("Iavarone_ID", "GarofanoClass.y"))
df[, cols_to_scale] <- scale(df[, cols_to_scale])

# Step 2: Split data into training and test sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(df$GarofanoClass.y, p = 0.7, list = FALSE)
train_data_samples <- df[train_index, ]
train_data <- data.frame(train_data_samples[1:34])  # Limit to first 53 features
test_data_samples <- df[-train_index, ]
test_data <- data.frame(test_data_samples[1:34])

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

# Step 5: Multiple Algorithms Applied

# 1. Logistic Regression Tuning
logistic_grid <- expand.grid(alpha = c(0, 0.5, 1),  # alpha = 0 (Ridge), 1 (Lasso), 0.5 (Elastic Net)
                             lambda = c(0.001, 0.01, 0.1, 1, 10))  # Regularization strength

logistic_model <- train(as.formula(paste("GarofanoClass.y", "~ .")),
                        data = train_data[, c(ranked_features, "GarofanoClass.y")],
                        method = "glmnet",
                        metric = "ROC",
                        tuneGrid = logistic_grid,  # Grid of hyperparameters
                        trControl = control)

# 2. Random Forest Tuning
rf_grid <- expand.grid(mtry = c(1, 3, 5, 7, 9))  # Example grid of 'mtry' values for tuning

rf_model <- train(as.formula(paste("GarofanoClass.y", "~ .")),
                  data = train_data[, c(ranked_features, "GarofanoClass.y")],
                  method = "rf",
                  metric = "ROC",
                  tuneGrid = rf_grid,  # Grid of hyperparameters
                  trControl = control)

# 3. SVM Radial Tuning
svm_grid <- expand.grid(sigma = c(0.01, 0.1, 1), C = c(1, 10, 100))  # Grid for radial SVM

svm_model <- train(as.formula(paste("GarofanoClass.y", "~ .")),
                   data = train_data[, c(ranked_features, "GarofanoClass.y")],
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

xgb_model <- train(as.formula(paste("GarofanoClass.y", "~ .")),
                   data = train_data[, c(ranked_features, "GarofanoClass.y")],
                   method = "xgbTree",
                   metric = "ROC",
                   tuneGrid = xgb_grid,  # Grid of hyperparameters
                   trControl = control)

# Evaluation: Compute ROC for cross-validated predictions and test set for each model
library(pROC)

# Function to compute and plot ROC for cross-validation
plot_roc <- function(model, test_data, ranked_features) {
  cv_predictions <- model$pred
  roc_curve <- roc(cv_predictions$obs, cv_predictions$Glycolytic)
  plot(roc_curve, legacy.axes = TRUE, print.auc = TRUE, main = paste("Cross Validation ROC -", model$method))
  
  # Test set ROC
  test_probabilities <- predict(model, newdata = test_data[, c(ranked_features, "GarofanoClass.y")], type = "prob")
  test_roc_curve <- roc(test_data$GarofanoClass.y, test_probabilities$Glycolytic)
  plot(test_roc_curve, legacy.axes = TRUE, print.auc = TRUE, main = paste("Test Set ROC -", model$method))
}

# Plot ROC for each model
plot_roc(logistic_model, test_data, ranked_features)
plot_roc(rf_model, test_data, ranked_features)
plot_roc(svm_model, test_data, ranked_features)
plot_roc(xgb_model, test_data, ranked_features)

# combined plot
# Function to compute ROC for cross-validation
compute_roc <- function(model, test_data, ranked_features) {
  # Cross-Validation ROC
  cv_predictions <- model$pred
  roc_curve_cv <- roc(cv_predictions$obs, cv_predictions$Glycolytic)  # Replace 'CE' with positive class
  
  # Test set ROC
  test_probabilities <- predict(model, newdata = test_data[, c(ranked_features, "GarofanoClass.y")], type = "prob")
  roc_curve_test <- roc(test_data$GarofanoClass.y, test_probabilities$Glycolytic)
  
  return(list(cv = roc_curve_cv, test = roc_curve_test))
}

# Compute ROC curves for all models
roc_logistic <- compute_roc(logistic_model, test_data, ranked_features)
roc_rf <- compute_roc(rf_model, test_data, ranked_features)
roc_svm <- compute_roc(svm_model, test_data, ranked_features)
roc_xgb <- compute_roc(xgb_model, test_data, ranked_features)

# Plot all ROC curves on one plot, adjusting AUC positions to avoid overlap
plot(roc_logistic$test, col = "blue", legacy.axes = TRUE, main = "Test Set ROC Comparison (Glycolytic)", 
     print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.5, lwd = 2)  # Logistic Regression

plot(roc_rf$test, add = TRUE, col = "red", 
     print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.45, lwd = 2)  # Random Forest

plot(roc_svm$test, add = TRUE, col = "green", 
     print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.4, lwd = 2)  # SVM

plot(roc_xgb$test, add = TRUE, col = "purple", 
     print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.35, lwd = 2)  # XGBoost

# Add a legend to the plot
legend("bottomright", legend = c("Logistic Regression", "Random Forest", "SVM", "XGBoost"),
       col = c("blue", "red", "green", "purple"), lwd = 2)

# EVENT GENERATION FOR DATA (GLYCOLYTIC) ----
library(tidyverse)
library(tibble)
library(gsubfn)
library(dendextend)

get_img_events <- function(img_df) {
  
  # Dissimilarity matrix based on euclidean distance
  d <- dist(img_df, method = "euclidean")
  # Hierarchical clustering using median linkage
  hc1 <- hclust(d, method = "median" )
  # expression rate dendrogram
  dend <- as.dendrogram(hc1)
  # Return all the nodes in the tree with corresponding samples
  events <- partition_leaves(dend)
  
  return(events)
}
#all data into neo4j
data_neo4j <- rbind(train_data_samples, test_data_samples)
data_neo4j <- data_neo4j[, c(ranked_features, "GarofanoClass.y", "Iavarone_ID")]
data_neo4j_optimal <- data_neo4j[, c(ranked_features, "GarofanoClass.y", "Iavarone_ID")]

img_names <- setdiff(colnames(data_neo4j), c("GarofanoClass.y", "Iavarone_ID"))
img_names_optimal <- setdiff(colnames(data_neo4j_optimal), c("GarofanoClass.y", "Iavarone_ID"))

# ALL SAMPLES W ALL FEATURES
setwd("/Users/m254284/Desktop/Neo4j_Projects/Img_events/ML/GCN_GARO/Glycolytic")

for (img_no in img_names) {
  # Print the img name and the number
  print(paste("Working on img no.", img_no))
  
  # Get the img data
  img_df <- as.data.frame(data_neo4j[img_no])
  rownames(img_df) <- data_neo4j$Iavarone_ID
  img_name <- colnames(img_df)
  
  # Get the img events
  img_events <- get_img_events(img_df = img_df)
  
  # Create an empty data frame to append the result of each img tree event
  df <- data.frame(matrix(ncol = 11, nrow = 0))  # Adjusted number of columns
  
  for (i in seq(1, length(img_events))) {
    # Get event characteristics
    event <- i
    
    # Get event patients (samples)
    unlisted_patients <- unlist(img_events[i])
    patients <- paste(unlisted_patients, collapse=",")
    
    # Get median img signal for the event
    patients_ge <- img_df %>% filter(row.names(img_df) %in% unlisted_patients)
    median_ge <- median(patients_ge[[1]])
    
    # Get the number of patients and if there is only one patient, mark as a leaf
    no_of_patients <- length(unlisted_patients)
    leaf_status <- ifelse(no_of_patients == 1, 1, 0)
    
    # Determine subtype
    tissue_annotations <- data_neo4j %>%
      filter(Iavarone_ID %in% unlisted_patients) %>%
      select(GarofanoClass.y)
    
    # Calculate the percentage of NE and CE samples
    percentage_glycolytic <- sum(tissue_annotations$GarofanoClass.y == "Glycolytic") / no_of_patients * 100
    percentage_other <- sum(tissue_annotations$GarofanoClass.y == "Other") / no_of_patients * 100
    
    # Tissue region can be recorded as "All" if we want to generalize
    tissue_region <- "All"
    
    # Append the event data to the dataframe
    df <- rbind(df, c(event, patients, median_ge, no_of_patients, img_name, 'img_value', leaf_status, tissue_region, percentage_glycolytic, percentage_other))
  }
  
  # Provide column names (adjusted for the new columns)
  colnames(df) <- c('event', 'samples', 'median_signal', 'no_of_patients', 'img_name', 'molecular_data', 'leaf_status', 'tissue_region', 'percentage_glycolytic', 'percentage_other')
  
  # Save each img's event data to a CSV file
  file_name <- paste(img_name, ".csv", sep="")
  write.csv(df, file=file_name, row.names = FALSE)
}

