import anthropic
from abc import ABC, abstractmethod
from src.config import ANTHROPIC_API_KEY, DEFAULT_MODEL, MAX_TOKENS, MAX_AGENT_ITERATIONS


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = model
        self.conversation_history = []
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Each agent defines its own personality and instructions."""
        pass
    
    @property
    @abstractmethod
    def tools(self) -> list[dict]:
        """Each agent defines which tools it can use."""
        pass
    
    @abstractmethod
    async def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute a tool call and return the result as a string.
        Each agent implements its own tool execution logic.
        """
        pass
    
    async def run(self, task: str) -> str:
        """
        The core agentic loop. This is where the magic happens.
        
        Flow:
        1. Send the task to Claude with available tools
        2. If Claude responds with text only → we're done
        3. If Claude responds with tool_use → execute the tool
        4. Feed the tool result back to Claude
        5. Repeat until Claude gives a final text answer or we hit max iterations
        
        Args:
            task: The user's request in natural language.
            
        Returns:
            The agent's final text response.
        """
        # Start fresh conversation with the user's task
        self.conversation_history = [
            {"role": "user", "content": task}
        ]
        
        for iteration in range(MAX_AGENT_ITERATIONS):
            # Call Claude with the full conversation history
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                tools=self.tools,
                messages=self.conversation_history,
            )
            
            
            if response.stop_reason == "end_turn":
                # Extract the final text from the response
                return self._extract_text(response)
            
            elif response.stop_reason == "tool_use":
                # Claude wants to use a tool. The response contains both
                # text (Claude's reasoning) and tool_use blocks.
                
                # Add Claude's full response to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content  # includes both text and tool_use
                })
                
                # Process each tool call in the response
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  🔧 Calling tool: {block.name}({block.input})")
                        
                        # Execute the tool and capture the result
                        result = await self.execute_tool(block.name, block.input)
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,  # must match the tool_use block
                            "content": result,
                        })
                
                # Feed tool results back to Claude
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results,
                })
            
            else:
                # Unexpected stop reason
                return self._extract_text(response)
        
        return "Error: Agent reached maximum iterations without producing a final answer."
    
    def _extract_text(self, response) -> str:
        """Extract text content from a Claude API response."""
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts) if text_parts else "No response generated."