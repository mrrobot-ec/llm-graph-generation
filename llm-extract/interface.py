# import gradio as gr
# from neo4j import GraphDatabase
# import pandas as pd
# import matplotlib.pyplot as plt
# import networkx as nx
# from io import BytesIO
# from PIL import Image

# # Neo4j database connection details
# uri = "bolt://localhost:7688"  # Adjust if your Neo4j server is running elsewhere
# username = "neo4j"
# password = "12345678"

# # Connect to Neo4j database
# driver = GraphDatabase.driver(uri, auth=(username, password))
# session = driver.session()

# # Function to execute Cypher queries
# def execute_query(query):
#     result = session.run(query)
#     return result

# # Function to get patient diagnoses
# def get_patient_diagnoses(patient_id):
#     query = f"""
#     MATCH (patient:Entity {{id: "{patient_id}"}})-[:RELATIONSHIP{{type: "has"}}]->(diagnosis:Entity)
#     RETURN diagnosis.id AS diagnosis_id
#     """
#     result = execute_query(query)
#     diagnoses = [{"Diagnosis ID": record["diagnosis_id"]} for record in result]
#     return pd.DataFrame(diagnoses)

# # Function to get patient medications
# def get_patient_medications(patient_id):
#     query = f"""
#     MATCH (patient:Entity {{id: "{patient_id}"}})-[:RELATIONSHIP{{type: "receives"}}]->(medication:Entity)
#     RETURN medication.id AS medication_id
#     """
#     result = execute_query(query)
#     medications = [{"Medication ID": record["medication_id"]} for record in result]
#     return pd.DataFrame(medications)

# # Function to get all medications in the database
# def get_all_medications():
#     query = """
#     MATCH (patient:Entity)-[:RELATIONSHIP {type: "receives"}]->(medication:Entity)
#     RETURN medication.id AS medication_id
#     """
#     result = execute_query(query)
#     medications = [{"Medication ID": record["medication_id"]} for record in result]
#     return pd.DataFrame(medications)

# # Function to create and display the subject-relationship-object (S-R-O) graph for the selected query type
# def plot_sro_graph(patient_id, query_type):
#     G = nx.Graph()  # Initialize the graph

#     if query_type == "Patient Diagnoses":
#         # Get patient diagnoses (Subject - Relationship - Object)
#         query_diagnoses = f"""
#         MATCH (patient:Entity {{id: "{patient_id}"}})-[:RELATIONSHIP{{type: "has"}}]->(diagnosis:Entity)
#         RETURN diagnosis.id AS diagnosis_id
#         """
#         result_diagnoses = execute_query(query_diagnoses)
#         for record in result_diagnoses:
#             patient_node = patient_id
#             diagnosis_node = record["diagnosis_id"]
#             G.add_edge(patient_node, diagnosis_node, relationship="has")

#     elif query_type == "Patient Medications":
#         # Get patient medications (Subject - Relationship - Object)
#         query_medications = f"""
#         MATCH (patient:Entity {{id: "{patient_id}"}})-[:RELATIONSHIP{{type: "receives"}}]->(medication:Entity)
#         RETURN medication.id AS medication_id
#         """
#         result_medications = execute_query(query_medications)
#         for record in result_medications:
#             patient_node = patient_id
#             medication_node = record["medication_id"]
#             G.add_edge(patient_node, medication_node, relationship="receives")

#     elif query_type == "All Medications in Database":
#         # Get all medications (Subject - Relationship - Object)
#         query_all_medications = """
#         MATCH (patient:Entity)-[:RELATIONSHIP {type: "receives"}]->(medication:Entity)
#         RETURN medication.id AS medication_id
#         """
#         result_all_medications = execute_query(query_all_medications)
#         for record in result_all_medications:
#             medication_node = record["medication_id"]
#             G.add_node(medication_node)

#     # Create plot for the graph
#     plt.figure(figsize=(12, 8))
#     pos = nx.spring_layout(G, seed=42)  # Position the nodes using a spring layout
#     labels = nx.get_edge_attributes(G, "relationship")
#     nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#     plt.title(f"Subject-Relationship-Object Graph for Patient {patient_id} ({query_type})")

#     # Save plot to a PIL Image
#     buf = BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     pil_image = Image.open(buf)
#     buf.close()

#     return pil_image

# # Gradio Interface to interact with the user
# def interface(patient_id, query_type):
#     # Generate S-R-O graph
#     # graph_image = plot_sro_graph(patient_id, query_type)

#     # Display the corresponding query results as a DataFrame
#     if query_type == "Patient Diagnoses":
#         df = get_patient_diagnoses(patient_id)
#     elif query_type == "Patient Medications":
#         df = get_patient_medications(patient_id)
#     elif query_type == "All Medications in Database":
#         df = get_all_medications()
#     else:
#         df = pd.DataFrame()

#     # return df, graph_image
#     return df

# # Gradio interface
# iface = gr.Interface(
#     fn=interface,
#     inputs=[
#         gr.Textbox(label="Patient ID (e.g., Patient ID: 10000980)"),
#         gr.Dropdown(choices=["Patient Diagnoses", "Patient Medications", "All Medications in Database"], label="Query Type"),
#         gr.Button(value="All")
#     ],
#     outputs=[gr.DataFrame()],
#     live=True
# )

# iface.launch()

import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from neo4j import GraphDatabase
import litellm
import os
from dotenv import load_dotenv
import re
import json
load_dotenv()

# 1. LLM Wrapper
class LiteLLMWrapper(LLM):
    def _call(self, prompt, stop=None):
        response = litellm.completion(
            model="ollama/deepseek-r1:32b",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response['choices'][0]['message']['content']
        print("LLM Response:", text)
        cypher = re.search(r"```cypher\n(.*?)```", text, re.DOTALL)
        if cypher is not None:
            print(cypher.group(1))
            return cypher.group(1)

        return "Error"
    @property
    def _llm_type(self):
        return "custom_litellm"

# 2. Neo4j driver
driver = GraphDatabase.driver(
    "bolt://localhost:7688",
    auth=("neo4j", "12345678")
)

def run_cypher(query):
    with driver.session() as session:
        result = session.run(query)
        return [dict(r) for r in result]

# 3. LangChain Setup
prompt_template = """
You are a helpful assistant that converts questions into Cypher queries for a Neo4j graph.
Assume that all the patients in this knowledge database have diabetes.
The Neo4j graph has the following relationships types:
- HAS_DIAGNOSIS
- PRESENTS_WITH
- RECEIVED_MEDICATION
- UNDERWENT_PROCEDURE
- HAS_ALLERGY
- DISCHARGE_STATUS
- HAS_ADMISSION_TYPE
- HAS_AGE
- HAS_GENDER
- HAS_ETHNICITY
- HAS_INSURANCE
- LIVES_WITH
- RESPONDED_TO
- HAS_VITAL_SIGN
The relationship types can be like the following data:
    {{
        "subject": "10000980",
        "relationship": "HAS_ETHNICITY",
        "object": "Race: BLACK/AFRICAN AMERICAN"
    }},
    {{
        "subject": "10000980",
        "relationship": "HAS_ADMISSION_TYPE",
        "object": "Admission Type: OBSERVATION ADMIT"
    }},
    {{
        "subject": "10000980",
        "relationship": "HAS_DIAGNOSIS",
        "object": "Diagnosis: type 2 diabetes mellitus with diabetic chronic kidney disease (ICD Code: E1122)"
    }},
    {{
        "subject": "10000980",
        "relationship": "RECEIVED_MEDICATION",
        "object": "Medication: Insulin"
    }},
    {{
        "subject": "10000980",
        "relationship": "PRESENTS_WITH",
        "object": "Chief Complaint: shortness of breath"
    }},
    {{
        "subject": "10000980",
        "relationship": "PRESENTS_WITH",
        "object": "several days of shortness of breath"
    }},
    {{
        "subject": "10000980",
        "relationship": "HAS_VITAL_SIGN",
        "object": "Weight gain over the past week (7lbs)"
    }},
    {{
        "subject": "10000980",
        "relationship": "RECEIVED_MEDICATION",
        "object": "Torsemide 40mg qd"
    }},
    {{
        "subject": "10000980",
        "relationship": "HAS_ALLERGY",
        "object": "No known allergies"
    }}
Use the information from above to struct the queries and return them to the user.
Here is the user's question:

{question}

Respond ONLY with the Cypher query and nothing else see the examples:
1. This would return when we want to know all the medications related to diabetes:
```cypher
MATCH p=()-[:RECEIVED_MEDICATION]->() RETURN p;
```
2. This would be the return also when we want to know all the medications related to diabetes, but with all DISTINCT:
```cypher
MATCH ()-[:RECEIVED_MEDICATION]->(drug) RETURN DISTINCT drug AS Drugs;
```
I want those types of queries, simples and concises. Nothing to fancy don't forget to add the ; at the end of each query.
Don't add any text explaining the query, I just need you to return the query nothing else.
"""

template = PromptTemplate.from_template(prompt_template)
chain = template | LiteLLMWrapper()

# 4. Main Gradio Function
def query_graph(question):
    try:
        cypher = chain.invoke({"question": question})
        result = run_cypher(cypher)
        print('resultado',result)
        return cypher, result
    except Exception as e:
        return "Error generating or executing Cypher", {"error": str(e)}

# 5. Gradio Interface
iface = gr.Interface(
    fn=query_graph,
    inputs=gr.Textbox(lines=2, placeholder="Ask something like: List all drugs for diabetes"),
    outputs=[
        gr.Textbox(label="Generated Cypher Query"),
        gr.JSON(label="Query Result from Neo4j")
    ],
    title="Neo4j Natural Language Query",
    description="Ask questions and get Cypher-powered results from your Neo4j knowledge graph."
)

iface.launch()


