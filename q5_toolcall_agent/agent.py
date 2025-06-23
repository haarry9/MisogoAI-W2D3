import os
import subprocess
import sys
from typing import Any, Dict, List
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class PythonExecTool(BaseTool):
    """Tool for executing Python code and returning stdout."""
    
    name: str = "python_exec"
    description: str = """Execute Python code and return the output. 
    Use this for counting characters, arithmetic calculations, or any computation that requires precision.
    Input should be valid Python code as a string."""
    
    def _run(self, code: str) -> str:
        """Execute Python code and return stdout."""
        try:
            # Create a temporary Python script
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute the Python script and capture output
            result = subprocess.run([sys.executable, temp_file], 
                                  capture_output=True, text=True, timeout=10)
            
            # Clean up temp file
            os.unlink(temp_file)
            
            if result.stderr:
                return f"Error: {result.stderr}"
            
            return result.stdout.strip() if result.stdout.strip() else "No output"
            
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out"
        except Exception as e:
            return f"Error: {str(e)}"


class NoOpTool(BaseTool):
    """Tool that does nothing - used for normal responses without computation."""
    
    name: str = "noop"
    description: str = """Use this tool when you want to provide a normal response without any computation.
    This tool does nothing and returns nothing."""
    
    def _run(self, query: str = "") -> str:
        """Do nothing and return empty string."""
        return ""


class ToolCallingAgent:
    """Agent that can call tools for computation or respond normally."""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the agent with tools and LLM."""
        
        # Set up OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key is required")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4", 
            temperature=0,
            max_tokens=1000
        )
        
        # Initialize tools
        self.tools = [PythonExecTool(), NoOpTool()]
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant with access to Python execution capabilities.
            
When users ask questions that require precise counting, character analysis, or arithmetic calculations, use the python_exec tool to get accurate results.

When users ask general questions that don't require computation, use the noop tool or respond normally.

Examples of when to use python_exec:
- Counting characters in strings
- Counting substring occurrences  
- Arithmetic calculations
- Mathematical computations

Be concise and accurate in your responses."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create agent
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def query(self, user_input: str) -> Dict[str, Any]:
        """Process user query and return response with tool call information."""
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return {
                "user_input": user_input,
                "response": response["output"],
                "success": True
            }
        except Exception as e:
            return {
                "user_input": user_input,
                "response": f"Error: {str(e)}",
                "success": False
            }


def main():
    """Main function to demonstrate the agent."""
    # Initialize agent (you'll need to set OPENAI_API_KEY environment variable)
    try:
        agent = ToolCallingAgent()
        print("Tool-Calling Agent initialized successfully!")
        print("You can now ask counting or arithmetic questions.")
        print("Type 'quit' to exit.\n")
        
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() == 'quit':
                break
                
            result = agent.query(user_input)
            print(f"Agent: {result['response']}\n")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OPENAI_API_KEY environment variable.")


if __name__ == "__main__":
    main() 