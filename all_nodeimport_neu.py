from py2neo import Graph

# Connect to your Neo4j database
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# Path to your CSV folder
csv_folder = "/Users/M254284/Desktop/Neo4j_Projects/Img_events/ML/GCN_GARO/Neuronal"

# Iterate through CSV files in the folder
import os
for filename in os.listdir(csv_folder):
    # Cypher query to load CSV file into Neo4j
    cypher_query = f"""
    LOAD CSV WITH HEADERS FROM 'file:///Neuronal/{filename}' AS row
    MERGE (n:Img {{img_name: row.img_name, event: row.event, tissue_region: row.tissue_region}})
    SET n.event = row.event,
        n.samples = row.samples,
        n.median_signal = row.median_signal,
        n.no_of_samples = row.no_of_patients,
        n.img_name = row.img_name,
        n.molecular_data = row.molecular_data,
        n.leaf_status = row.leaf_status,
        n.tissue_region = row.tissue_region,
        n.quartile = row.quartile,
        n.percentage_neuronal = row.percentage_neuronal,
        n.percentage_other = row.percentage_other
    """

    # Execute the Cypher query
    graph.run(cypher_query)
