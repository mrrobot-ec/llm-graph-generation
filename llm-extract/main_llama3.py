#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from litellm import completion
from typing import List
import json
import litellm
from tqdm import tqdm
# litellm.set_verbose=True


# In[ ]:


response = completion(
    # model="ollama/llama3.2",
    model="ollama/adrienbrault/nous-hermes2pro-llama3-8b:q8_0",
    messages=[{ "content": f"You are a name entity recognition system you will extract entities related to medicine, specially for diabetes information that I will give you in the following prompt"
               f"and you will extract all meaningful entities from the given discharge summary"
               f"You will return only JSON outputs","role": "user"}],
    # json=True
)
print(response)


# In[11]:


def format_entities(ent_list:List[str]) -> str:
    return "\n\n".join([e for e in ent_list])


# In[12]:


import json

with open('diabetes_data_preprocessed.json') as f:
    chunks = json.load(f)
print(chunks[0])


# In[15]:


system_message = """
Extract all meaningful entities from the given discharge summary, focusing on the following topics:
- Patient Demographics and Identifiers
- Primary and Secondary Diagnoses
- Medications and Treatments (both at admission and discharge)
- Discharge Instructions
- Additional Notes (including tests, mental status, and other relevant observations)

Return all extracted entities as a JSON array. Each object in the array should represent a meaningful entity and contain two fields:
1. "entity": The extracted entity (e.g., names, conditions, medications, dosages, instructions, dates, etc.)
2. "type": The type of the entity (e.g., "name", "diagnosis", "medication", "dosage", "instruction", "test", "mental status", etc.)

Ensure each entity is unique, meaningful, and relevant to the discharge summary.

Example output:
[
  {"entity": "10000980", "type": "Patient_ID"},
  {"entity": "BLACK/AFRICAN AMERICAN", "type": "Patient_race"},
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
  # api_key=OPENAI_API_KEY,
  model="ollama/adrienbrault/nous-hermes2pro-llama3-8b:q8_0",
  # model="ollama/llama3.2:latest",
  # model="gpt-3.5-turbo",
  messages=[{"content": system_message,"role": "system"}, {"content": user_message.format(text=text),"role": "user"}],
  max_tokens=7000,
  temperature=0.3,
  # format = "json",
  api_base="http://localhost:11434"
)
print(response.choices[0].message.content)
# triples = json.loads(response.choices[0].message.content)
# triples


# In[17]:


system_message = """
Extract all the relationships between the following entities ONLY based on the given context.
Break the Discharge Summary: into smaller pieces of triplets and remove any '\' if any triplet contains it.
Return a list of JSON objects. For example:

[{"subject": "Patient ID: 10000980",
  "relationship": "is",
  "object": "Race: BLACK/AFRICAN AMERICAN"},
  {"subject": "Patient ID: 10000980",
  "relationship": "has",
  "object": "Admission Type: OBSERVATION ADMIT"},
  {"subject": "Patient ID: 10000980",
  "relationship": "has",
  "object": "Diagnosis: type 2 diabetes mellitus with diabetic chronic kidney disease (ICD Code: E1122)"},
  {"subject": "Patient ID: 10000980",
  "relationship": "receives",
  "object": "Medication: Insulin"},
  {"subject": "Patient ID: 10000980",
  "relationship": "has", "object":
  "Chief Complaint: shortness of breath"},
  {"subject": "Patient ID: 10000980",
  "relationship": "presents with",
  "object": "several days of shortness of breath"},
  {"subject": "Patient ID: 10000980",
  "relationship": "has",
  "object": "Weight gain over the past week (7lbs)"},
  {"subject": "Patient ID: 10000980",
  "relationship": "is taking",
  "object": "Torsemide 40mg qd"},
  {"subject": "Patient ID: 10000980",
  "relationship": "has no known",
  "object": "Allergies"},]


- ONLY return triples and nothing else. None of 'subject', 'relationship' and 'object' can be empty.

 """

text = chunks[0]['context']

user_message = "Context: {text}\n\nTriples:"
response = completion(
  # api_key=OPENAI_API_KEY,
  model="ollama/adrienbrault/nous-hermes2pro-llama3-8b:q8_0",
  # model="ollama/llama3.2:latest",
  # model="gpt-3.5-turbo",
  messages=[{"content": system_message,"role": "system"}, {"content": user_message.format(text=text),"role": "user"}],
  max_tokens=7500,
  # format = "json",
  api_base="http://localhost:11434",
  temperature=0.3
)
# print(response.choices[0].message.content)
triples = json.loads(response.choices[0].message.content)
print(triples)
len(triples)


# In[ ]:


import time

errors = []
all_triples = []

# for i in tqdm(range(2)):
for i in tqdm(range(len(chunks))):
    try:
        text = chunks[i]['context']

        user_message = "Context: {text}\n\nTriples:"
        response = completion(
        # api_key=OPENAI_API_KEY,
        model="ollama/adrienbrault/nous-hermes2pro-llama3-8b:q8_0",
        # model="ollama/llama3.2:latest",
        # model="gpt-3.5-turbo",
        messages=[{"content": system_message,"role": "system"}, {"content": user_message.format(text=text),"role": "user"}],
        max_tokens=7500,
        # format = "json",
        api_base="http://localhost:11434",
        temperature=0.3
        )
        triples = json.loads(response.choices[0].message.content)
        all_triples.append(triples)
        time.sleep(3)
    except Exception as e:
        print(f"Error for chunk {i}, {e}")
        errors.append(response.choices[0].message.content)
        all_triples.append([])


# In[ ]:


output_file = "triples.json"
json_data = json.dumps(all_triples, indent=4)
with open(output_file, "w") as file:
    file.write(json_data)


# In[ ]:


input_file = "triples.json"
with open(input_file, "r") as file:
    all_triples = json.load(file)

all_triples[0]


# In[ ]:




