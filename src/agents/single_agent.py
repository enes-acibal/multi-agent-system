import asyncio
import sys
from src.agents.base_agent import BaseAgent
from src.tools.web_search import web_search, WEB_SEARCH_TOOL
from src.tools.file_reader import read_file, FILE_READER_TOOL


class SingleAgent(BaseAgent):
    """A general-purpose agent with search and file reading abilities."""
    
    @property
    def system_prompt(self) -> str:
        return """You are a helpful research assistant. You have access to tools
that let you search the web and read local files. 

When given a task:
1. Think about what information you need
2. Use your tools to gather that information
3. Synthesize what you found into a clear, well-structured answer

Always cite your sources when using web search results. If you can't find 
reliable information, say so honestly rather than guessing.

Be concise but thorough. Prefer structured answers with clear sections 
when the topic is complex."""
    
    @property
    def tools(self) -> list[dict]:
        return [WEB_SEARCH_TOOL, FILE_READER_TOOL]
    
    async def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Route tool calls to their implementations."""
        if tool_name == "web_search":
            return await web_search(**tool_input)
        elif tool_name == "read_file":
            return read_file(**tool_input)
        else:
            return f"Error: Unknown tool '{tool_name}'"


# ─── Entry point for testing ─────────────────────────────────────────
async def main():
    """Run the agent from the command line."""
    if len(sys.argv) < 2:
        task = "What are the latest developments in AI agents and multi-agent systems?"
    else:
        task = " ".join(sys.argv[1:])
    
    print(f"📋 Task: {task}")
    print("=" * 60)
    
    agent = SingleAgent()
    result = await agent.run(task)
    
    print("\n" + "=" * 60)
    print(f"📝 Result:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())