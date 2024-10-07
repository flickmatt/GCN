# mRMRe for Imaging and Neo4j Prep
library(caret)
library(mRMRe)
# DATA SPLIT AND MRMR ----
# Select columns that contain "Raw_Mean" but not "nii.gz"
df_cols <- grep("Raw_Mean", names(master), value = TRUE)
df_cols <- df_cols[!grepl("nii.gz", df_cols)]
df_cols <- df_cols[!grepl("SPGRC", df_cols)]

# Add the 'MRI.contrast.enhancing.annotation' column to the selection
df_cols <- c(df_cols, "MRI.contrast.enhancing.annotation", "Iavarone_ID")

# Subset the dataframe
df <- na.omit(master[, df_cols])

# Identify the columns to scale (excluding "MRI.contrast.enhancing.annotation")
cols_to_scale <- setdiff(df_cols, c("MRI.contrast.enhancing.annotation", "Iavarone_ID"))

# Scale the columns
df[, cols_to_scale] <- scale(df[, cols_to_scale])

# The "MRI.contrast.enhancing.annotation" column remains unchanged
# Step 1: Split the data into training and test sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(df$MRI.contrast.enhancing.annotation, p = 0.7, list = FALSE)
train_data_samples <- df[train_index, ]
train_data <- train_data_samples[1:53]
# train_data <- train_data_samples[1:54] # with t1c
test_data_samples <- df[-train_index, ]
test_data <- test_data_samples[1:53]
# test_data <- test_data_samples[1:54] # with t1c

# Set the target index (assuming the last column is the target)
target_index <- ncol(train_data)

# Convert the target column to an ordered factor
train_data[, target_index] <- as.ordered(train_data[, target_index])

# Convert all feature columns (except the target) to numeric where possible
train_data[-target_index] <- lapply(train_data[-target_index], function(col) {
  if (is.character(col)) {
    # Attempt to convert to numeric
    as.numeric(as.character(col))
  } else if (is.factor(col)) {
    # Convert unordered factors to ordered factors
    as.ordered(col)
  } else {
    col
  }
})

# Check column types to ensure all are numeric or ordered factors
sapply(train_data, class)

# Prepare the mRMRe dataset
mrmr_data <- mRMR.data(data = train_data)

# Run mRMR with the correct target index
mrmr_result <- mRMR.classic(data = mrmr_data, target_indices = target_index, feature_count = 53)

# ITERATIVE LOG REG W POS FEATURES ----
# Ensure necessary libraries are loaded
library(mRMRe)
library(glmnet)  # Optional for cross-validation, not used directly in this case

# Assuming you already have 'df' and 'mrmr_result' from previous steps

# Step 1: Filter features with positive mRMR scores
feature_scores <- mrmr_result@scores[[1]]  # Scores of selected features
selected_features <- mrmr_result@filters[[1]]  # Indices of selected features
positive_score_indices <- selected_features[feature_scores > 0]  # Only positive-scoring features
ranked_features <- colnames(df)[positive_score_indices]  # Feature names for the positive features

# Target column
target_column <- "MRI.contrast.enhancing.annotation"

# Initialize a vector to store model AIC values
aic_values <- numeric(length(ranked_features))

# Step 2: Iterate over the number of top features to use, starting from the end (most important)
for (k in seq_along(ranked_features)) {
  # Always include the most important feature(s) at the end of the ranked list
  selected_features <- c(ranked_features[(length(ranked_features) - k + 1):length(ranked_features)], target_column)
  
  # Subset the dataframe to include only the selected features
  selected_df <- train_data[, selected_features]
  
  # Ensure the target column is treated as a factor
  selected_df[, target_column] <- as.factor(selected_df[, target_column])
  
  # Fit the logistic regression model
  model <- glm(as.formula(paste(target_column, "~ .")), family = binomial, data = selected_df)
  
  # Step 3: Store the AIC value for the model
  aic_values[k] <- AIC(model)
}

# Step 4: Find the number of features that gives the model with the lowest AIC
optimal_k <- which.min(aic_values)
all_k <- length(aic_values)

# Print the best number of features
cat("Optimal number of features:", optimal_k, "\n")
cat("AIC for the optimal model:", aic_values[optimal_k], "\n")

# Step 5: Refit the logistic regression model using the optimal number of features
# Always include the most important features at the end, OR ALL FEATURES (SECOND)
optimal_features <- ranked_features[(length(ranked_features) - optimal_k + 1):length(ranked_features)]
all_features <- ranked_features[(length(ranked_features) - all_k + 1):length(ranked_features)]
#optimal_features <- ranked_features[1:length(ranked_features)]
final_model <- glm(as.formula(paste(target_column, "~ .")), 
                   family = binomial, 
                   data = train_data[, c(all_features, target_column)])

# Output the summary of the final model with the optimal features
summary(final_model)

# PREDICTION
predicted_probs <- predict(final_model, newdata = test_data, type = "response")
predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)  # Adjust threshold if necessary

# Generate confusion matrix and calculate metrics on test data
conf_matrix <- table(Predicted = predicted_classes, Actual = test_data$MRI.contrast.enhancing.annotation)
print(conf_matrix)

# Calculate accuracy and other metrics
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
precision <- conf_matrix[2,2] / sum(conf_matrix[2,])
recall <- conf_matrix[2,2] / sum(conf_matrix[,2])
f1_score <- 2 * ((precision * recall) / (precision + recall))

cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
cat("Accuracy:", accuracy, "\n")

library(pROC)
roc_curve <- roc(test_data$MRI.contrast.enhancing.annotation, predicted_probs)
auc(roc_curve)  # Print AUC value
plot(roc_curve)  # Plot ROC curve

# CROSS VALIDATION W/O ITERATIVE LOG RES ----
# Required Libraries
library(mRMRe)
library(caret)
library(glmnet)
library(pROC)
library(ggplot2)

# Select columns that contain "Raw_Mean" but not "nii.gz"
df_cols <- grep("Raw_Mean", names(master), value = TRUE)
df_cols <- df_cols[!grepl("nii.gz", df_cols)]
df_cols <- df_cols[!grepl("SPGRC", df_cols)]

# Add the 'MRI.contrast.enhancing.annotation' column to the selection
df_cols <- c(df_cols, "MRI.contrast.enhancing.annotation", "Iavarone_ID")

# Subset the dataframe
df <- na.omit(master[, df_cols])

# Identify the columns to scale (excluding "MRI.contrast.enhancing.annotation")
cols_to_scale <- setdiff(df_cols, c("MRI.contrast.enhancing.annotation", "Iavarone_ID"))

# Scale the columns
df[, cols_to_scale] <- scale(df[, cols_to_scale])

# Step 1: Split the data into training and test sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(df$MRI.contrast.enhancing.annotation, p = 0.7, list = FALSE)
train_data_samples <- df[train_index, ]
train_data <- data.frame(train_data_samples[1:53])  # Limit to first 53 features
test_data_samples <- df[-train_index, ]
test_data <- data.frame(test_data_samples[1:53])

# Set the target index (assuming the last column is the target)
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

# mRMR feature selection
mrmr_data <- mRMR.data(data = train_data)
mrmr_result <- mRMR.classic(data = mrmr_data, target_indices = target_index, feature_count = 53)

# Filter features with positive mRMR scores
feature_scores <- mrmr_result@scores[[1]]
selected_features <- mrmr_result@filters[[1]]
positive_score_indices <- selected_features[feature_scores > 0]
ranked_features <- colnames(df)[positive_score_indices]

# Cross-Validation Setup
control <- trainControl(method = "cv", 
                        number = 5, 
                        classProbs = TRUE,  # To get probabilities for ROC curve
                        summaryFunction = twoClassSummary, 
                        savePredictions = TRUE)  # Save predictions to create ROC curve later

# Refit model using optimal features OR ALL FEATURES
final_model <- train(as.formula(paste("MRI.contrast.enhancing.annotation", "~ .")),
                     data = train_data[, c(ranked_features, "MRI.contrast.enhancing.annotation")],
                     method = "glm",
                     family = "binomial",
                     metric = "ROC",  # Optimize based on ROC metric
                     trControl = control)

# Extract predictions from cross-validation
cv_predictions <- final_model$pred

library(pROC)
library(ggplot2)

# Compute ROC curve using cross-validated predictions
roc_curve <- roc(cv_predictions$obs, cv_predictions$CE)  # 'CE' is the probability of positive class
roc_plot <- plot(roc_curve, legacy.axes = T, main = "Cross Validation ROC of Logistic Regression", print.auc = T)


# EVENT GENERATION FOR TRAINING DATA ----
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
data_neo4j <- data_neo4j[, c(ranked_features, "MRI.contrast.enhancing.annotation", "Iavarone_ID")]
data_neo4j_optimal <- data_neo4j[, c(optimal_features, "MRI.contrast.enhancing.annotation", "Iavarone_ID")]

img_names <- setdiff(colnames(data_neo4j), c("MRI.contrast.enhancing.annotation", "Iavarone_ID"))
img_names_optimal <- setdiff(colnames(data_neo4j_optimal), c("MRI.contrast.enhancing.annotation", "Iavarone_ID"))

# ALL SAMPLES W ALL FEATURES
setwd("/Users/m254284/Desktop/Neo4j_Projects/Img_events/ML/GCN/All_NoT1C")

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
    
    # Determine tissue region (NE vs CE)
    tissue_annotations <- data_neo4j %>%
      filter(Iavarone_ID %in% unlisted_patients) %>%
      select(MRI.contrast.enhancing.annotation)
    
    # Calculate the percentage of NE and CE samples
    percentage_NE <- sum(tissue_annotations$MRI.contrast.enhancing.annotation == "NE") / no_of_patients * 100
    percentage_CE <- sum(tissue_annotations$MRI.contrast.enhancing.annotation == "CE") / no_of_patients * 100
    
    # Tissue region can be recorded as "All" if we want to generalize
    tissue_region <- "All"
    
    # Append the event data to the dataframe
    df <- rbind(df, c(event, patients, median_ge, no_of_patients, img_name, 'img_value', leaf_status, tissue_region, percentage_NE, percentage_CE))
  }
  
  # Provide column names (adjusted for the new columns)
  colnames(df) <- c('event', 'samples', 'median_signal', 'no_of_patients', 'img_name', 'molecular_data', 'leaf_status', 'tissue_region', 'percentage_NE', 'percentage_CE')
  
  # Save each img's event data to a CSV file
  file_name <- paste(img_name, ".csv", sep="")
  write.csv(df, file=file_name, row.names = FALSE)
}

# ALL SAMPLES W OPTIMAL FEATURES
setwd("/Users/m254284/Desktop/Neo4j_Projects/Img_events/ML/GCN/All_Optimal")

for (img_no in img_names_optimal) {
  # Print the img name and the number
  print(paste("Working on img no.", img_no))
  
  # Get the img data
  img_df <- as.data.frame(data_neo4j_optimal[img_no])
  rownames(img_df) <- data_neo4j_optimal$Iavarone_ID
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
    
    # Determine tissue region (NE vs CE)
    tissue_annotations <- data_neo4j_optimal %>%
      filter(Iavarone_ID %in% unlisted_patients) %>%
      select(MRI.contrast.enhancing.annotation)
    
    # Calculate the percentage of NE and CE samples
    percentage_NE <- sum(tissue_annotations$MRI.contrast.enhancing.annotation == "NE") / no_of_patients * 100
    percentage_CE <- sum(tissue_annotations$MRI.contrast.enhancing.annotation == "CE") / no_of_patients * 100
    
    # Tissue region can be recorded as "All" if we want to generalize
    tissue_region <- "All"
    
    # Append the event data to the dataframe
    df <- rbind(df, c(event, patients, median_ge, no_of_patients, img_name, 'img_value', leaf_status, tissue_region, percentage_NE, percentage_CE))
  }
  
  # Provide column names (adjusted for the new columns)
  colnames(df) <- c('event', 'samples', 'median_signal', 'no_of_patients', 'img_name', 'molecular_data', 'leaf_status', 'tissue_region', 'percentage_NE', 'percentage_CE')
  
  # Save each img's event data to a CSV file
  file_name <- paste(img_name, ".csv", sep="")
  write.csv(df, file=file_name, row.names = FALSE)
}
