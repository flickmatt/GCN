# GCN
- Comparing graph convolutional network to logistic regression for regional GBM classification using MRI features.
- Neo4j graph database contains nodes for each event of hierarchically clustered samples (non-enhancing (NE) and contrast-enhancing (CE) samples together) across MRI features selected using mRMRe filtering for NE/CE differentiation. All positive scoring features on mRMRe were included in the graph. 
- Edges were generated between any nodes containing at least one common sample. Cypher query included in repository.
- Hyperparameter tuning performed using optuna for prediction of 100% NE or 100% CE nodes.
- Line 202 begins to evaluate the hyperparameter tuned GCN.
- ROC based on 5-fold cross validation is used to evaluate GCN and compare to logistic regression. 

# mRMR_GLM_NECE
- The most recent method used to create the logistic regression model for comparison to GCN.
- All positive scoring features on mRMRe were included in the model. No further filtering using AIC was performed. 
- ROC based on 5-fold cross validation is used to evaluated and compare to GCN.
- Samples them clustered using hierarchical clustering for Neo4j nodes on the same MRI features. 

# EdgeQuery
- Neo4j query used to create edges.

# all_nodeimport
- Script to import hierarchical clustering events as nodes in Neo4j across all MRI features with positive mRMRe output.
