import asyncio


# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import litellm
from litellm import experimental_mcp_client
import json
litellm._turn_on_debug()
server_params = StdioServerParameters(
    command="/home/andres/.local/bin/uvx",  # Adjust path to your venv Python
    args=[ "mcp-neo4j-cypher",
        "--db-url",
        "bolt://localhost:7688",
        "--username",
        "neo4j",
        "--password",
        "12345678"],  # or your entrypoint module
    env=None
)
message_context = """ You are an expert that converts insightfull questions into Cypher into insightfull queries for a Neo4j graph
Assume that all the patients in this knowledge database have diabetes.
This knowledge graph is loaded with diabetes patient information from the dataset MIMIC IV the The Neo4j graph has the following relationships types:
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
also the following triplets were loaded into the neo4j so here is an example on how does the graph could look like
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
"""
async def run_mcp_litellm_interaction():

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await experimental_mcp_client.load_mcp_tools(session=session, format="mcp")
            print("MCP TOOLS: ", tools)
            messages = [{"role": "user", "content": message_context}]
            llm_response = await litellm.acompletion(
                 model="ollama/deepseek-r1:32b",
                messages=messages,
                tools=tools,
            )
            messages = [{"role": "user", "content": "I want to know what medications are commonly prescribed for diabetes patients"}]
            llm_response = await litellm.acompletion(
                 model="ollama/deepseek-r1:32b",
                messages=messages,
                tools=tools,
            )
            print("LLM RESPONSE: ", json.dumps(llm_response, indent=4, default=str))

            openai_tool = llm_response["choices"][0]["message"]["tool_calls"][0]

            # Call the tool using MCP client
            call_result = await experimental_mcp_client.call_openai_tool(
                session=session,
                openai_tool=openai_tool,
            )
            print("MCP TOOL CALL RESULT: ", call_result)

            # send the tool result to the LLM
            messages.append(llm_response["choices"][0]["message"])
            messages.append(
                {
                    "role": "tool",
                    "content": str(call_result.content[0].text),
                    "tool_call_id": openai_tool["id"],
                }
            )
            print("final messages with tool result: ", messages)
            llm_response = await litellm.acompletion(
                 model="ollama/deepseek-r1:32b",
                messages=messages,
                tools=tools,
            )
            print(
                "FINAL LLM RESPONSE: ", json.dumps(llm_response, indent=4, default=str)
            )

if __name__ == "__main__":
    asyncio.run(run_mcp_litellm_interaction())