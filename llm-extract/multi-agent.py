import asyncio
from google.genai import types
from google.adk.runners import Runner
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.llm_agent import LlmAgent, Agent
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)
import litellm
# litellm._turn_on_debug()


async def get_tools_async():
    """Gets tools from the File System MCP Server."""
    print("Attempting to connect to MCP Neo4j server...")
    tools, exit_stack = await MCPToolset.from_server(
        # Use StdioServerParameters for local process communication
        connection_params=StdioServerParameters(
            command="/home/andres/.local/bin/uvx",  # Adjust path to your venv Python
            args=["mcp-neo4j-cypher",
                  "--db-url",
                  "bolt://localhost:7688",
                  "--username",
                  "neo4j",
                  "--password",
                  "12345678"],  # or your entrypoint module
            env=None
        )
    )
    print("MCP Toolset created successfully.")
    return tools, exit_stack

model = LiteLlm(
    model="openai/mistral-small3.1",
    api_base="http://localhost:11434/v1",
    api_key="asdf"
)

root_model = LiteLlm(
    model="ollama_chat/deepseek-r1:32b",
)


async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
        print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:  # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            break  # Stop processing events once the final response is found

    print(f"<<< Agent Response: {final_response_text}")


async def get_agent_async():
    """Creates an ADK Agent equipped with tools from the MCP Server and sub-agents to get the schema database and
    build the query and run the query."""
    tools, exit_stack = await get_tools_async()
    print(f"Fetched {len(tools)} tools from MCP server: ")
    print(f"Fetched {len(tools)} tools from MCP server. {tools[0].name}")
    print(f"Fetched {len(tools)} tools from MCP server. {tools[1].name}")
    print(f"Fetched {len(tools)} tools from MCP server. {tools[2].name}")

    neo4j_run_query = LlmAgent(
        model=model,  # Adjust model name if needed based on availability
        name='run_query_agent',
        description="Sub-agent in charge to execute the queries into the Neo4j database",
        instruction="You are an agent that retrieves information from Neo4j"
                    "You will run the queries using the read_neo4j_cypher tool",
        tools=[tools[1]],  # Provide the MCP tools to the ADK agent
    )

    neo4j_get_schema = LlmAgent(
        model=model,  # Adjust model name if needed based on availability
        name='get_schema_agent',
        description="Sub-agent in charge to get the schema from the Neo4j database",
        instruction="You are an agent that gets the schema from Neo4j"
                    "You will get the schema using the get_neo4j_schema tool",
        tools=[tools[0]],  # Provide the MCP tools to the ADK agent
    )

    agent_team = Agent(
        model=root_model,  # Adjust model name if needed based on availability
        name='neo4j_assistant',
        description="The main coordinator at. Handles human languaje requests and delegates schema/queries to specialists.",
        instruction="You are a helpful assistant that converts human questions into Cypher queries for a Neo4j graph."
                    "Assume that all the patients in this knowledge database have diabetes."
                    "The knowledge graph is loaded with diabetes patient information from the dataset MIMIC IV."
                    "1. You must do always have to retrieve the schema of the knowledge database using the neo4j_get_schema sub-agent"
                    "2. After getting all the context, you will build simple queries to achieve the users goal"
                    "Keep in mind that the labels are always :Entity be generic with that you cannot use something like (d:Drug)"
                    "Also keep in mind that you always have to refer to the nodes as d.id, because that is where the information is,"
                    "don't do d.NAME or d.type or any other form or you can do return d or return d.id"
                    "You cannot do this type of queries {label: 'drug'} only like e:Entity"
                    "You have specialized sub-agents: "
                    "1. You will get the schema always the schema before building any query using the neo4j_get_schema sub-agent"
                    "2. You will run the queries using the neo4j_run_query sub-agent"
                    "you have to try at least 5 types of queries before giving to the user that no information were found",
        sub_agents=[neo4j_run_query, neo4j_get_schema]
        # tools=tools
    )
    print(
        f"✅ Root Agent '{agent_team.name}' created using model '{root_model}' with sub-agents: {[sa.name for sa in agent_team.sub_agents]}")

    return agent_team, exit_stack


# async def async_main():
#     session_service = InMemorySessionService()
#     artifacts_service = InMemoryArtifactService()
#     APP_NAME = "neo4j_query_app_t"
#     USER_ID = "neo4j_query_t"
#     SESSION_ID = "session_001_tools_test"  # Using a fixed ID for simplicity

#     session = session_service.create_session(
#         state={}, app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
#     )

#     query = "Get me the schema from the database"
#     print(f"User Query: '{query}'")
#     content = types.Content(role='user', parts=[types.Part(text=query)])

#     root_agent, exit_stack = await get_agent_async()

#     runner = Runner(
#         app_name=APP_NAME,
#         agent=root_agent,
#         artifact_service=artifacts_service,  # Optional
#         session_service=session_service,
#     )

#     print("Running agent...")
#     events_async = runner.run_async(
#         session_id=session.id, user_id=session.user_id, new_message=content
#     )

#     # await call_agent_async(query, runner, USER_ID, SESSION_ID)
#     async for event in events_async:
#         print(f"Event received: {event}")
#         print("\n\n")
#         print(event.content.parts)
#         print(event.content.parts[0].text)

#     # Crucial Cleanup: Ensure the MCP server process connection is closed.
#     print("Closing MCP server connection...")
#     await exit_stack.aclose()
#     print("Cleanup complete.")


async def run_team_conversation():
    print("\n--- Testing Agent Team Delegation ---")
    # InMemorySessionService is simple, non-persistent storage for this tutorial.
    session_service = InMemorySessionService()
    artifacts_service = InMemoryArtifactService()
    # Define constants for identifying the interaction context
    APP_NAME = "neo4j_query_app_t"
    USER_ID = "neo4j_query_t"
    SESSION_ID = "session_001_tools_test"  # Using a fixed ID for simplicity

    # Create the specific session where the conversation will happen
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    # --- Get the actual root agent object ---
    # Use the determined variable name
    # actual_root_agent = globals()['agent_team']
    root_agent, exit_stack = await get_agent_async()
    # Create a runner specific to this agent team test
    runner_agent_team = Runner(
        agent=root_agent,  # Use the root agent object
        app_name=APP_NAME,  # Use the specific app name
        artifact_service=artifacts_service,
        session_service=session_service  # Use the specific session service
    )
    # Corrected print statement to show the actual root agent's name
    print(f"Runner created for agent '{root_agent.name}'.")

    # Always interact via the root agent's runner, passing the correct IDs
    await call_agent_async(query="Get me the schema from the database",
                           runner=runner_agent_team,
                           user_id=USER_ID,
                           session_id=SESSION_ID)
    await call_agent_async(query="Can you give me an overview of the medicine that is present in the database?",
                           runner=runner_agent_team,
                           user_id=USER_ID,
                           session_id=SESSION_ID)
    # await call_agent_async(query="List useful information about the demographics/ethnicity in the database?",
    #                        runner=runner_agent_team,
    #                        user_id=USER_ID,
    #                        session_id=SESSION_ID)

    # Crucial Cleanup: Ensure the MCP server process connection is closed.
    print("Closing MCP server connection...")
    await exit_stack.aclose()
    print("Cleanup complete.")


if __name__ == '__main__':
    try:
        # asyncio.run(async_main())
        asyncio.run(run_team_conversation())
    except Exception as e:
        print(f"An error occurred: {e}")
