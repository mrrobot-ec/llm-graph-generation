import gradio as gr
from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from PIL import Image

# Neo4j database connection details
uri = "bolt://localhost:7687"  # Adjust if your Neo4j server is running elsewhere
username = "neo4j"
password = "12345678"

# Connect to Neo4j database
driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()

# Function to execute Cypher queries
def execute_query(query):
    result = session.run(query)
    return result

# Function to get patient diagnoses
def get_patient_diagnoses(patient_id):
    query = f"""
    MATCH (patient:Entity {{id: "{patient_id}"}})-[:RELATIONSHIP{{type: "has"}}]->(diagnosis:Entity)
    RETURN diagnosis.id AS diagnosis_id
    """
    result = execute_query(query)
    diagnoses = [{"Diagnosis ID": record["diagnosis_id"]} for record in result]
    return pd.DataFrame(diagnoses)

# Function to get patient medications
def get_patient_medications(patient_id):
    query = f"""
    MATCH (patient:Entity {{id: "{patient_id}"}})-[:RELATIONSHIP{{type: "receives"}}]->(medication:Entity)
    RETURN medication.id AS medication_id
    """
    result = execute_query(query)
    medications = [{"Medication ID": record["medication_id"]} for record in result]
    return pd.DataFrame(medications)

# Function to get all medications in the database
def get_all_medications():
    query = """
    MATCH (patient:Entity)-[:RELATIONSHIP {type: "receives"}]->(medication:Entity)
    RETURN medication.id AS medication_id
    """
    result = execute_query(query)
    medications = [{"Medication ID": record["medication_id"]} for record in result]
    return pd.DataFrame(medications)

# Function to create and display the subject-relationship-object (S-R-O) graph for the selected query type
def plot_sro_graph(patient_id, query_type):
    G = nx.Graph()  # Initialize the graph

    if query_type == "Patient Diagnoses":
        # Get patient diagnoses (Subject - Relationship - Object)
        query_diagnoses = f"""
        MATCH (patient:Entity {{id: "{patient_id}"}})-[:RELATIONSHIP{{type: "has"}}]->(diagnosis:Entity)
        RETURN diagnosis.id AS diagnosis_id
        """
        result_diagnoses = execute_query(query_diagnoses)
        for record in result_diagnoses:
            patient_node = patient_id
            diagnosis_node = record["diagnosis_id"]
            G.add_edge(patient_node, diagnosis_node, relationship="has")

    elif query_type == "Patient Medications":
        # Get patient medications (Subject - Relationship - Object)
        query_medications = f"""
        MATCH (patient:Entity {{id: "{patient_id}"}})-[:RELATIONSHIP{{type: "receives"}}]->(medication:Entity)
        RETURN medication.id AS medication_id
        """
        result_medications = execute_query(query_medications)
        for record in result_medications:
            patient_node = patient_id
            medication_node = record["medication_id"]
            G.add_edge(patient_node, medication_node, relationship="receives")

    elif query_type == "All Medications in Database":
        # Get all medications (Subject - Relationship - Object)
        query_all_medications = """
        MATCH (patient:Entity)-[:RELATIONSHIP {type: "receives"}]->(medication:Entity)
        RETURN medication.id AS medication_id
        """
        result_all_medications = execute_query(query_all_medications)
        for record in result_all_medications:
            medication_node = record["medication_id"]
            G.add_node(medication_node)

    # Create plot for the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Position the nodes using a spring layout
    labels = nx.get_edge_attributes(G, "relationship")
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(f"Subject-Relationship-Object Graph for Patient {patient_id} ({query_type})")

    # Save plot to a PIL Image
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    pil_image = Image.open(buf)
    buf.close()

    return pil_image

# Gradio Interface to interact with the user
def interface(patient_id, query_type):
    # Generate S-R-O graph
    # graph_image = plot_sro_graph(patient_id, query_type)

    # Display the corresponding query results as a DataFrame
    if query_type == "Patient Diagnoses":
        df = get_patient_diagnoses(patient_id)
    elif query_type == "Patient Medications":
        df = get_patient_medications(patient_id)
    elif query_type == "All Medications in Database":
        df = get_all_medications()
    else:
        df = pd.DataFrame()

    # return df, graph_image
    return df

# Gradio interface
iface = gr.Interface(
    fn=interface,
    inputs=[
        gr.Textbox(label="Patient ID (e.g., Patient ID: 10000980)"),
        gr.Dropdown(choices=["Patient Diagnoses", "Patient Medications", "All Medications in Database"], label="Query Type")
    ],
    outputs=[gr.DataFrame()],
    live=True
)

iface.launch()
