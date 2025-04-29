import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Optional
from contextlib import AsyncExitStack
from ollama import Client

class OllamaMCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = Client()
        self.tools = []


    async def connect_to_server(self):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
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

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.tools = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    },
                } for tool in response.tools]
        print("\nConnected to server with tools:", [tool["function"]["name"] for tool in self.tools])

    def get_tools(self):
        return self.tools

    async def process_query(self, query: str) -> str:
        """Process a query using LLM and available tools"""
        schema = await self.session.call_tool("get_neo4j_schema")
        print("------ help",schema.content)
        system_context: str = f"""
        You are a helpful assistant that converts questions into Cypher queries for a Neo4j graph.
        Assume that all the patients in this knowledge database have diabetes.
        The knowledge graph is loaded with diabetes patient information from the dataset MIMIC IV.
        The schema of the knowledge database is: {schema.content[0].text}
        Describe the schema
        Don't use p:Patient or any other because we are using Entity to represent the nodes,
        only information from the schema to generate the cypher query.
        Don't use exists or any other non supported property Cypher shell version 1.1.15.
        Use the tool read-neo4j-cypher to execute the query
        Use the information from above to make the queries and execute them and return some insightful information from the query.
        Don't add information or fake data to the response only return from the query.
        Don't ever tell me this: Please execute the query using your Neo4j instance to get the actual results. You are supposed to give me the information from my neo4j.
        """


        messages = [
            {"role": "system", "content": system_context},
            {
                "role": "user",
                "content": query
            }
        ]

        response = self.client.chat(
            model="qwen2.5",
            messages=messages,
            tools=self.tools,
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []
        print("--------", response)
        if response.message.content:
            final_text.append(response.message.content)
        elif response.message.tool_calls:
            for tool in response.message.tool_calls:
                tool_name = tool.function.name
                tool_args = tool.function.arguments

                # Execute tool call
                result = await self.session.call_tool(tool_name, dict(tool_args))
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                messages.append({
                    "role": "user",
                    "content": result.content[0].text
                })

                response = self.client.chat(
                    model="qwen2.5",
                    messages=messages,
                )

                final_text.append(response.message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    client = OllamaMCPClient()
    print("client initiated")
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
# query to the graph using the tools to give me a list of drugs used for diabetes