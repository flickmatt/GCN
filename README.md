# GCN
- Comparing graph convolutional network to logistic regression for regional GBM classification using MRI features.
- Neo4j graph database contains nodes for each event of hierarchically clustered samples (non-enhancing (NE) and contrast-enhancing (CE) samples together) across MRI features selected using mRMRe filtering for NE/CE differentiation. All positive scoring features on mRMRe were included in the graph. 
- Edges were generated between any nodes containing at least one common sample. Cypher query included in repository.
- ROC based on 5-fold cross validation is used to evaluate GCN and compare to logistic regression. 

# mRMR_GLM_NECE
- The most recent method used to create the logistic regression model for comparison to GCN.
- All positive scoring features on mRMRe were included in the model. No further filtering using AIC was performed. 
- ROC based on 5-fold cross validation is used to evaluated and compare to GCN.
- Samples them clustered using hierarchical clustering for Neo4j nodes on the same MRI features. 
