#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from litellm import completion
from typing import List
import json
import litellm
import litellm.timeout
from tqdm import tqdm
import re
# litellm.request_timeout = 20000
# litellm.set_verbose=True
# litellm._turn_on_debug()


# In[ ]:


# sk-d676ec249f3e44468eb9264fe0736705

# curl -X POST -H "Authorization: Bearer sk-d676ec249f3e44468eb9264fe0736705" -H 'Content-Type: application/json' http://130.68.125.206:3000/ollama/api/generate -d '{ "model": "deepseek-r1:32b", "prompt": "Why is the sky blue?", "stream": false }'


# In[ ]:


response = completion(
    # model="ollama/llama3.2",
    model="ollama/deepseek-r1:32b",
    messages=[{ "content": f"You are a name entity recognition system you will extract entities related to medicine, specially for diabetes information that I will give you in the following prompt"
               f"and you will extract all meaningful entities from the given discharge summary"
               f"You will return only JSON outputs","role": "user"}],
    # json=True
)
print(response)


# In[ ]:


def format_entities(ent_list:List[str]) -> str:
    return "\n\n".join([e for e in ent_list])


# In[ ]:


import json

with open('diabetes_data_preprocessed.json') as f:
    chunks = json.load(f)
print(chunks[0])


# In[ ]:


system_message = """
Extract all meaningful entities from the given discharge summary, focusing on the following topics:
- Patient Demographics and Identifiers
- Primary and Secondary Diagnoses
- Medications and Treatments (both at admission and discharge)

Return all extracted entities as a JSON array. Each object in the array should represent a meaningful entity and contain two fields:
1. "entity": The extracted entity (e.g., names, conditions, medications, dosages, instructions, dates, etc.)
2. "type": The type of the entity (e.g., "name", "diagnosis", "medication", "dosage", "instruction", "test", "mental status", etc.)

Ensure each entity is unique, meaningful, and relevant to the discharge summary.

Example output:
[
  {"entity": "10000980", "type": "patient_id"},
  {"entity": "BLACK/AFRICAN AMERICAN", "type": "patient_race"},
  {"entity": "Acute on Chronic Diastolic Congestive Heart Failure", "type": "diagnosis"},
  {"entity": "Hypertension", "type": "diagnosis"},
  {"entity": "Allopurinol", "type": "medication"},
  {"entity": "80 mg", "type": "dosage"},
  {"entity": "qpm", "type": "dosage frequency"},
  {"entity": "Monitor and limit salt intake", "type": "discharge_instruction"},
  {"entity": "clear and coherent", "type": "mental_status"},
  {"entity": "chest X-ray", "type": "test"}
]

Return **strictly JSON format** for all extracted entities.

 """

text = chunks[11]['context']

user_message = "Context: {text}\n\nTriples:"
response = completion(
  model="ollama/deepseek-r1:32b",
  messages=[{"content": system_message,"role": "system"}, {"content": user_message.format(text=text),"role": "user"}],
  temperature=0,
)
text = response.choices[0].message.content
json_match = re.search(r'json\n(\[.*\])\n', text, re.DOTALL)
if json_match is not None:
  print(json.loads(json_match.group(1)))


# In[ ]:


import re

# The given string
text = """
<think>
Alright, I need to extract all meaningful entities from the given discharge summary. The user has specified focusing on three main areas: Patient Demographics and Identifiers, Primary and Secondary Diagnoses, and Medications and Treatments at both admission and discharge.

First, I'll start by identifying the patient's demographics. Looking through the text, I see "Patient ID: 10001176" which is straightforward. The race is mentioned as "WHITE", so that's another entity. The sex is listed as "f", which stands for female. There's also a mention of "morbid obesity" under past medical history, but since the user wants to focus on demographics, I'll include it here.

Next, moving on to diagnoses. The primary diagnosis seems to be "diabetes mellitus without mention of complication type ii or unspecified type not stated as uncontrolled". There's also a note about "presumptive pna", which stands for pneumonia. Additionally, the patient has a history of "coronary artery disease" and "morbid obesity", so these should be included as secondary diagnoses.

For medications and treatments, at admission, the patient was given "levofloxacin 750mg iv". At discharge, she received "Ativan 2mg po", "Tylenol 2g", and "Zofran 4mg". I'll list each medication separately with their respective dosages.

I need to ensure that each entity is unique and relevant. I should avoid including non-essential details like test results or physical exam findings unless they're directly related to the specified categories. Also, I must structure each entry as a JSON object with "entity" and "type".

Let me double-check if I've covered all areas: patient ID, race, sex, primary diagnosis, secondary diagnoses, admission medications, and discharge medications. Everything seems accounted for.

Finally, I'll format the extracted entities into a JSON array, making sure each entry is correctly labeled with its type.
</think>

```json
[
   {"entity": "10001176",  "type": "Patient_Identifier"},
   {"entity": "WHITE",  "type": "Patient_race"},
   {"entity": "f",  "type": "sex"},
   {"entity": "diabetes mellitus without mention of complication type ii or unspecified type not stated as uncontrolled",  "type": "Primary_diagnosis"},
   {"entity": "presumptive pna",  "type": "Secondary_diagnosis"},
   {"entity": "coronary artery disease",  "type": "Secondary_diagnosis"},
   {"entity": "morbid obesity",  "type": "Secondary_diagnosis"},
   {"entity": "Tylenol 2g",  "type": "medication_discharge"},
   {"entity": "Zofran 4mg",  "type": "medication_discharge"}
]
```
"""
text = response.choices[0].message.content
json_match = re.search(r'json\n(\[.*\])\n', text, re.DOTALL)
if json_match is not None:
  entities = json.loads(json_match.group(1))
  print(entities)


# In[ ]:


system_message = """
Extract all the relationships between the following entities ONLY based on the given context.
Break the Discharge Summary: into smaller pieces of triplets and remove any '\' if any triplet contains it.
Use only the following relationship types:
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
Return a list of JSON objects. For example:
[
    {
        "subject": "Patient ID: 10000980",
        "relationship": "HAS_ETHNICITY",
        "object": "Race: BLACK/AFRICAN AMERICAN"
    },
    {
        "subject": "Patient ID: 10000980",
        "relationship": "HAS_ADMISSION_TYPE",
        "object": "Admission Type: OBSERVATION ADMIT"
    },
    {
        "subject": "Patient ID: 10000980",
        "relationship": "HAS_DIAGNOSIS",
        "object": "Diagnosis: type 2 diabetes mellitus with diabetic chronic kidney disease (ICD Code: E1122)"
    },
    {
        "subject": "Patient ID: 10000980",
        "relationship": "RECEIVED_MEDICATION",
        "object": "Medication: Insulin"
    },
    {
        "subject": "Patient ID: 10000980",
        "relationship": "PRESENTS_WITH",
        "object": "Chief Complaint: shortness of breath"
    },
    {
        "subject": "Patient ID: 10000980",
        "relationship": "PRESENTS_WITH",
        "object": "several days of shortness of breath"
    },
    {
        "subject": "Patient ID: 10000980",
        "relationship": "HAS_VITAL_SIGN",
        "object": "Weight gain over the past week (7lbs)"
    },
    {
        "subject": "Patient ID: 10000980",
        "relationship": "RECEIVED_MEDICATION",
        "object": "Torsemide 40mg qd"
    },
    {
        "subject": "Patient ID: 10000980",
        "relationship": "HAS_ALLERGY",
        "object": "No known allergies"
    }
]

- ONLY return triples and nothing else. None of 'subject', 'relationship' and 'object' can be empty.

 """

text = chunks[0]['context']

user_message = "Context: {text}\n\nTriples:"
response = completion(
  model="ollama/deepseek-r1:32b",
  messages=[{"content": system_message,"role": "system"}, {"content": user_message.format(text=text),"role": "user"}],
  temperature=0
)
text = response.choices[0].message.content
json_match = re.search(r'json\n(\[.*\])\n', text, re.DOTALL)
if json_match is not None:
  triples = json.loads(json_match.group(1))
  print(triples)


# In[ ]:


import time

errors = []
all_triples = []

for i in tqdm(range(len(chunks))):
    try:
        text = chunks[i]['context']

        user_message = "Context: {text}\n\nTriples:"
        response = completion(
        model="ollama/deepseek-r1:32b",
        messages=[{"content": system_message,"role": "system"}, {"content": user_message.format(text=text),"role": "user"}],
        temperature=0
        )
        text = response.choices[0].message.content
        json_match = re.search(r'json\n(\[.*\])\n', text, re.DOTALL)
        if json_match is not None:
            triples = json.loads(json_match.group(1))
            print(triples)
        all_triples.append(triples)
        time.sleep(3)
    except Exception as e:
        print(f"Error for chunk {i}, {e}")
        errors.append(response.choices[0].message.content)
        all_triples.append([])


# In[ ]:


output_file = "triples-r1.json"
json_data = json.dumps(all_triples, indent=4)
with open(output_file, "w") as file:
    file.write(json_data)


# In[ ]:


input_file = "triples-r1.json"
with open(input_file, "r") as file:
    all_triples = json.load(file)

type(all_triples)
all_triples


# In[ ]:


import csv



# Step 1: Extract unique nodes
nodes = set()
for triple_list in all_triples:
    for triplet in triple_list:
        # Add "subject" if it exists
        if "subject" in triplet:
            nodes.add((triplet["subject"], "Entity"))
        # Add "object" if it exists
        if "object" in triplet:
            nodes.add((triplet["object"], "Entity"))

# Step 2: Write nodes.

with open("nodes-r1.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    writer.writerows(nodes)

# Step 3: Extract relationships
edges = []
for triple_list in all_triples:
    for triplet in triple_list:
        if "subject" in triplet and "object" in triplet and "relationship" in triplet:
            edges.append((triplet["subject"], triplet["object"], triplet["relationship"]))

# Step 4: Write edges.csv
with open("edges-r1.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target", "relationship"])
    writer.writerows(edges)

print("Nodes and edges files generated successfully.")


# In[ ]:


# sudo cp nodes.csv /var/lib/neo4j/import/
# sudo cp edges.csv /var/lib/neo4j/import/
# cypher-shell -u neo4j -p <your_password>
# LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
# CREATE (:Entity {id: row.id, label: row.label});
# LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
# MATCH (a:Entity {id: row.source}), (b:Entity {id: row.target})
# CREATE (a)-[:RELATIONSHIP {type: row.relationship}]->(b);
# ALL DATABASE
# MATCH (n)-[r]->(m) RETURN n, r, m;
# QUERY RELATIONSHIPS
# MATCH (n)-[r]->(m) RETURN r LIMIT 10;
# QUERY PATIENT ALL DIAGNOSES
# MATCH (patient:Entity {id: "Patient ID: 10000980"})-[:RELATIONSHIP{type: "has"}]->(diagnosis:Entity)
# RETURN diagnosis.id AS diagnosis_id;
# QUERY ALL THE MEDICATION
# MATCH (patient:Entity {id: "Patient ID: 10000980"})-[:RELATIONSHIP{type: "receives"}]->(medication:Entity)
# RETURN medication.id AS medication_id
# QUERY ALL THE MEDICATION IN THE DB
# MATCH (patient:Entity)-[:RELATIONSHIP {type: "receives"}]->(medication:Entity)
# RETURN medication.id AS medication_id;

