import os
import time
import asyncio
import logging
import streamlit as st
from google.genai import types
from google.adk.runners import Runner
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.llm_agent import LlmAgent, Agent
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

import warnings
warnings.filterwarnings("ignore")
import litellm
litellm._turn_on_debug()
logging.basicConfig(level=logging.INFO)  # Configure logging level (ERROR hides most ADK logs, INFO shows more detail)

# Define constants for identifying the interaction context
APP_NAME_INTERFACE = "neo4j_assistant_app"
USER_ID = "neo4j_assistant"
SESSION_ID = "session_001_neo4j_assistant"  # Using a fixed ID for simplicity
adk_session_key = ""


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
                  "12345678"],
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

async def get_agent_async():
    """Creates an ADK Agent equipped with tools from the MCP Server and sub-agents to get the schema database and
    build the query and run the query."""
    tools, exit_stack = await get_tools_async()
    print(f"Fetched {len(tools)} tools from MCP server: ")
    await exit_stack.__aenter__()
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

    return agent_team

import nest_asyncio

nest_asyncio.apply()

def get_agent_sync():
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(get_agent_async())

# --------------------------------------------------------------------------
# ADK Initialization and Runner Helper Functions
# --------------------------------------------------------------------------
# Use Streamlit's caching for resources (@st.cache_resource). This ensures that
# the ADK Runner and SessionService are initialized only once per user's
# browser session, maintaining state continuity across Streamlit script reruns.
@st.cache_resource
def initialize_adk():
    """
    Initializes the ADK Runner and InMemorySessionService for the application.
    Manages the unique ADK session ID within the Streamlit session state.
    Returns:
        tuple: (Runner instance, active ADK session ID)
    """
    print("--- ADK Init: Attempting to initialize Runner and Session Service... ---")
    # InMemorySessionService stores all session data (history, state dictionaries)
    # in the RAM of the process running the Streamlit app. Data is lost if the
    # Python process stops. For persistent storage, explore DatabaseSessionService.
    session_service = InMemorySessionService()
    print(f"--- ADK Init: InMemorySessionService instantiated. ---")


    root_agent = get_agent_sync()


    # The Runner connects the Agent definition with the SessionService to handle
    # the execution flow for each user interaction.
    runner = Runner(
        agent=root_agent,  # The agent configuration defined above
        app_name=APP_NAME_INTERFACE,  # Identifier for this runner instance
        session_service=session_service  # Service used to load/save session data
    )
    print(f"--- ADK Init: Runner instantiated for agent '{root_agent.name}'. ---")
    # We need a persistent session ID for the ADK conversation within the context
    # of a single user's interaction with the Streamlit app. We store this ID
    # in Streamlit's own session state (`st.session_state`).
    adk_session_key = 'adk_session_id_final_mem_v2'  # Unique key within st.session_state
    if adk_session_key not in st.session_state:
        # If this is the first time this Streamlit session is running initialize_adk,
        # generate a new, unique session ID for the ADK conversation.
        session_id = f"streamlit_session_final_mem_v2_{int(time.time())}_{os.urandom(4).hex()}"
        st.session_state[adk_session_key] = session_id  # Store the new ID in Streamlit's state
        print(f"--- ADK Init: Generated new ADK session ID: {session_id} ---")
        try:
            # Create the corresponding session record within the ADK SessionService.
            # This session starts with an empty state dictionary `{}`.
            session_service.create_session(
                app_name=APP_NAME_INTERFACE,
                user_id=USER_ID,
                session_id=session_id,
                state={}  # Initialize with an empty state
            )
            print(f"--- ADK Init: Successfully created new session in ADK SessionService. ---")
        except Exception as e:
            # Log and re-raise errors during initial session creation
            print(f"--- ADK Init: FATAL ERROR - Could not create initial session in ADK SessionService: {e} ---")
            logging.exception("ADK Session Service create_session failed:")
            raise  # Stop execution if the session can't be created
    else:
        # If adk_session_key exists in st.session_state, reuse the existing ADK session ID.
        session_id = st.session_state[adk_session_key]
        print(f"--- ADK Init: Reusing existing ADK session ID from Streamlit state: {session_id} ---")
        # **Important Check for InMemorySessionService**:
        # Since InMemorySessionService loses data if the script restarts (e.g., code change, server reboot),
        # we must verify if the session *actually* still exists in the service's memory.
        if not session_service.get_session(app_name=APP_NAME_INTERFACE, user_id=USER_ID, session_id=session_id):
            print(
                f"--- ADK Init: WARNING - Session {session_id} not found in InMemorySessionService memory (likely due to script restart). Recreating session. State will be lost. ---")
            try:
                # Recreate the session record in the service. The state will be reset to empty.
                session_service.create_session(
                    app_name=APP_NAME_INTERFACE,
                    user_id=USER_ID,
                    session_id=session_id,
                    state={}  # Recreated session starts with empty state
                )
            except Exception as e:
                # Handle errors during recreation attempt
                print(
                    f"--- ADK Init: ERROR - Could not recreate missing session {session_id} in ADK SessionService: {e} ---")
                logging.exception("ADK Session Service recreation failed:")
                # Depending on requirements, you might raise an error here or allow proceeding with a potentially inconsistent state.
    print(f"--- ADK Init: Initialization sequence complete. Runner is ready. Active Session ID: {session_id} ---")
    # Return the configured runner and the session ID to be used for interactions
    return runner, session_id


async def run_adk_async(runner: Runner, session_id: str, user_message_text: str) -> str:
    """
    Asynchronously executes one turn of the ADK agent conversation.
    Args:
        runner: The initialized ADK Runner.
        session_id: The current ADK session ID.
        user_message_text: The text input from the user for this turn.
    Returns:
        The agent's final text response as a string.
    """
    print(f"\n--- ADK Run: Starting async execution for session {session_id} ---")
    print(f"--- ADK Run: Processing User Query (truncated): '{user_message_text[:150]}...' ---")
    # Format the user's message into the google.genai.types.Content structure required by ADK runner.
    content = types.Content(
        role='user',  # Standard role identifier for user input
        parts=[types.Part(text=user_message_text)]  # The actual text content
    )
    final_response_text = "[Agent encountered an issue and did not produce a final response]"  # Default error message
    start_time = time.time()  # Start timing the agent execution
    # try:
        # The core ADK interaction: runner.run_async processes the new message within the session context.
        # It's an async generator, yielding Event objects that represent stages of the agent's turn
        # (e.g., planning, tool call request, tool result received, LLM response chunk, final response).
    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
        logging.info(f"[Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")
        # In this simple UI, we only need the agent's final output for the turn.
        # The `is_final_response()` method on the event identifies this.
        if event.is_final_response():
            print(f"--- ADK Run: Final response event received. ---")
            # Safely extract the text from the final event's content.
            # The content structure is Content -> parts (list) -> Part -> text.
            if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:  # Handle potential errors/escalations
                # Handle cases where the final event might not contain standard text
                # (e.g., an error occurred, or the agent structure is different).
                final_response_text = "[Agent finished but produced no text output]"
                print(f"--- ADK Run: WARNING - Final event received, but no text content found. Event: {event} ---")
            break  # Stop iterating through events once the final response is captured
        # --- Optional: Inspecting Intermediate Events ---
        # else:
        #     # You could log or handle other event types here for debugging or advanced UIs
        #     event_type = type(event).__name__
        #     author = getattr(event, 'author', 'N/A')
        #     print(f"--- ADK Run: Intermediate event received - Type: {event_type}, Author: {author} ---")
        #     # Example: Check for tool calls
        #     if hasattr(event, 'actions') and event.actions and hasattr(event.actions,
        #                                                                'function_call') and event.actions.function_call:
        #         print(f"--- ADK Run: -> Tool call requested: {event.actions.function_call.name} ---")
        #     # Example: Check for tool responses being processed
        #     if hasattr(event, 'actions') and event.actions and hasattr(event.actions,
        #                                                                'function_response') and event.actions.function_response:
        #         print(f"--- ADK Run: -> Processing tool response for ID: {event.actions.function_response.id} ---")
    # except Exception as e:
        # Catch any exceptions that occur during the runner.run_async execution.
        # print(f"--- ADK Run: !! EXCEPTION during agent execution: {e} !! ---")
        # logging.exception("ADK runner.run_async failed:")  # Log the full traceback
        # # Provide a user-friendly error message
        # final_response_text = f"Sorry, an error occurred while processing your request. Please check the logs or try again later. (Error: {e})"
    # Calculate and log the duration of the agent's turn
    end_time = time.time()
    duration = end_time - start_time
    print(f"--- ADK Run: Turn execution completed in {duration:.2f} seconds. ---")
    print(f"--- ADK Run: Final Response (truncated): '{final_response_text[:150]}...' ---")
    # Return the captured final response text
    return final_response_text


# Since Streamlit's main execution flow is synchronous, we need a helper
# function to call our asynchronous `run_adk_async` function.
def run_adk_sync(runner: Runner, session_id: str, user_message_text: str) -> str:
    """
    Synchronous wrapper that executes the asynchronous run_adk_async function.
    Uses asyncio.run() to manage the event loop.
    """
    # asyncio.run() creates a new event loop, runs the provided coroutine until
    # it completes, and then closes the event loop.
    return asyncio.run(run_adk_async(runner, session_id, user_message_text))


print("✅ ADK Runner initialization and helper functions defined.")

# --------------------------------------------------------------------------
# Streamlit User Interface Setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ADK assistant & Chat Agent",
    layout="wide",  # Use wide layout for more space
    initial_sidebar_state="auto"  # Keep sidebar visible initially
)
st.title("📰 News & Chat Assistant (Powered by ADK & Gemini)")
st.markdown("""
Interact with an AI agent that can fetch news from BBC/NPR or just chat.
**Examples:**
*   Ask for
`latest news` (gets past 7 days).
*   Request `news from YYYY-MM-DD` (e.g., `news from 2024-04-10`).
*   Use `news from today` or `news from yesterday`.
*   After a briefing, ask follow-up questions like `tell me more about the first item` or `what was the link for the NPR story?` (The agent uses its memory!).
*(Note: News feed history is typically limited to ~2 weeks)*
""")
st.divider()  # Add a visual separator
# --- Initialize ADK Runner and Session ---
# This block attempts to get the initialized ADK components.
# Thanks to @st.cache_resource on initialize_adk(), this runs only once
# per browser session unless the cache is cleared or the script changes significantly.
try:
    adk_runner, current_session_id = initialize_adk()
    # Display initialization success and part of the session ID in the sidebar
    st.sidebar.success(f"ADK Initialized\nSession: ...{current_session_id[-12:]}", icon="✅")
except Exception as e:
    # If ADK initialization fails (e.g., API error, configuration issue), display a critical error.
    st.error(f"**Fatal Error:** Could not initialize the ADK Runner or Session Service: {e}", icon="❌")
    st.error(
        "Please check the terminal logs for more details, ensure your API key is valid, and restart the application.")
    logging.exception("Critical ADK Initialization failed in Streamlit UI context.")
    st.stop()  # Stop the app if ADK fails to initialize

# --- Chat Interface Implementation ---
# Use Streamlit's session state to store the chat message history.
# This makes the chat history persist across reruns of the script triggered by UI interactions.
message_history_key = "messages_final_mem_v2"  # Use the same key consistently
if message_history_key not in st.session_state:
    # If no history exists for this session, initialize it as an empty list.
    st.session_state[message_history_key] = []
    print("Initialized Streamlit message history.")
# Display the existing chat messages from the history.
# This runs every time the script reruns (e.g., after user input).
# print(f"Displaying {len(st.session_state[message_history_key])} messages from history.")
for message in st.session_state[message_history_key]:
    # Use st.chat_message to render messages with appropriate icons (user/assistant).
    with st.chat_message(message["role"]):
        # Render message content using Markdown. Ensure HTML is not allowed for security.
        st.markdown(message["content"], unsafe_allow_html=False)
# Chat input field at the bottom of the page.
# `st.chat_input` returns the user's text when they press Enter or click Send.
if prompt := st.chat_input("Ask for news (e.g., 'latest news'), follow up, or just chat..."):
    print(f"User input received: '{prompt[:50]}...'")
    # 1. Append and display the user's message immediately.
    st.session_state[message_history_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=False)
    # 2. Process the user's prompt with the ADK agent and display the response.
    with st.chat_message("assistant"):
        # Use st.empty() as a placeholder to update with the full response later.
        # This gives a slightly better UX than just waiting and then showing the text.
        message_placeholder = st.empty()
        # Show a thinking indicator while the backend processes the request.
        with st.spinner("Assistant is thinking... "):
            try:
                # Call the synchronous wrapper function to run the ADK agent turn.
                agent_response = run_adk_sync(adk_runner, current_session_id, prompt)
                # Update the placeholder with the agent's complete response.
                message_placeholder.markdown(agent_response, unsafe_allow_html=False)
            except Exception as e:
                # If an error occurs during the ADK run, display it in the chat.
                error_msg = f"Sorry, an error occurred while processing your request: {e}"
                st.error(error_msg)  # Show error prominently in the chat UI
                agent_response = f"Error: Failed to get response. {e}"  # Store simplified error in history
                logging.exception("Error occurred within the Streamlit chat input processing block.")
    # 3. Append the agent's response (or error message) to the chat history.
    st.session_state[message_history_key].append({"role": "assistant", "content": agent_response})
    # Streamlit automatically reruns the script here, which redraws the chat history including the new messages.
    print("Agent response added to history. Streamlit will rerun.")

# --- Sidebar Information Display ---
# Add useful information to the sidebar for context/debugging.
st.sidebar.divider()
st.sidebar.header("Agent Details")
st.sidebar.caption(f"**Agent Name:** `{APP_NAME_INTERFACE}`")
st.sidebar.caption(f"**User ID:** `{USER_ID}`")
# Display the active ADK session ID (retrieve safely from st.session_state)
st.sidebar.caption(f"**Session ID:** `{st.session_state.get(adk_session_key, 'N/A')}`")
st.sidebar.caption(f"**LLM Model:** `{root_model.model}`")
st.sidebar.caption("Powered by Google Agent Development Kit.")
# Optional: Display raw state for debugging
with st.sidebar.expander("Show Raw ADK Session State"):
    try:
        current_session = adk_runner.session_service.get_session(app_name=APP_NAME_INTERFACE, user_id=USER_ID,
                                                                 session_id=current_session_id)
        if current_session:
            st.json(current_session.state)
        else:
            st.write("Session not found in service.")
    except Exception as e:
        st.error(f"Could not retrieve session state: {e}")

print("✅ Streamlit UI Rendering Complete.")
