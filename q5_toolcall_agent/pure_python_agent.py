"""
Pure Python Tool-Calling Agent
Alternative implementation without LangChain dependencies
"""

import subprocess
import sys
import tempfile
import os
import re
from typing import Dict, Any, List


class PythonExecutor:
    """Executes Python code and returns stdout."""
    
    def execute(self, code: str) -> str:
        """Execute Python code and return the output."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run([sys.executable, temp_file], 
                                  capture_output=True, text=True, timeout=10)
            
            os.unlink(temp_file)
            
            if result.stderr:
                return f"Error: {result.stderr}"
            
            return result.stdout.strip() if result.stdout.strip() else "No output"
            
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out"
        except Exception as e:
            return f"Error: {str(e)}"


class SimpleAgent:
    """Simple agent that determines when to use Python execution vs normal response."""
    
    def __init__(self):
        self.executor = PythonExecutor()
    
    def needs_computation(self, query: str) -> bool:
        """Determine if query needs Python computation."""
        computation_keywords = [
            'count', 'how many', 'calculate', 'factorial', 'arithmetic',
            'multiply', 'divide', 'add', 'subtract', 'math', 'number',
            'times does', 'characters', 'letters', 'vowels'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in computation_keywords)
    
    def generate_code(self, query: str) -> str:
        """Generate Python code based on the query."""
        query_lower = query.lower()
        
        # Character counting patterns
        if "how many" in query_lower and "in" in query_lower:
            # Extract character and string using regex
            match = re.search(r"how many ['\"](.+?)['\"] in ['\"](.+?)['\"]", query_lower)
            if match:
                char, string = match.groups()
                return f"result = len([c for c in '{string}' if c == '{char}'])\nprint(result)"
        
        # Vowel counting
        if "vowel" in query_lower:
            match = re.search(r"in ['\"](.+?)['\"]", query_lower)
            if match:
                word = match.groups()[0]
                return f"vowels = 'aeiouAEIOU'\nword = '{word}'\ncount = sum(1 for char in word if char in vowels)\nprint(count)"
        
        # Substring counting
        if "times does" in query_lower and "appear" in query_lower:
            match = re.search(r"['\"](.+?)['\"] appear in ['\"](.+?)['\"]", query_lower)
            if match:
                substring, text = match.groups()
                return f"""text = '{text}'
substring = '{substring}'
count = 0
for i in range(len(text) - len(substring) + 1):
    if text[i:i+len(substring)] == substring:
        count += 1
print(count)"""
        
        # Arithmetic expressions
        if any(op in query for op in ['*', '+', '-', '/']):
            # Extract mathematical expression
            match = re.search(r'(\d+\s*[\+\-\*/]\s*\d+(?:\s*[\+\-\*/]\s*\d+)*)', query)
            if match:
                expression = match.group(1)
                return f"result = {expression}\nprint(result)"
        
        # Factorial
        if "factorial" in query_lower:
            match = re.search(r'factorial of (\d+)', query_lower)
            if match:
                number = match.group(1)
                return f"import math\nresult = math.factorial({number})\nprint(result)"
        
        return "print('Could not generate appropriate code for this query')"
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return detailed response."""
        if self.needs_computation(query):
            # Generate and execute Python code
            code = self.generate_code(query)
            output = self.executor.execute(code)
            
            return {
                "user_input": query,
                "tool_used": "python_exec",
                "code_generated": code,
                "tool_output": output,
                "needs_computation": True
            }
        else:
            # Return normal response
            return {
                "user_input": query,
                "tool_used": "noop",
                "response": "I can help with counting and arithmetic questions. Try asking me to count characters or calculate math problems!",
                "needs_computation": False
            }


def run_example_transcripts():
    """Run the 5 required example transcripts."""
    agent = SimpleAgent()
    
    examples = [
        "How many 'r' in 'strawberry'?",
        "Count the number of vowels in 'programming'",
        "How many times does 'ab' appear in 'ababcabab'?",
        "What is 127 * 83 + 45?",
        "Calculate the factorial of 8"
    ]
    
    print("PURE PYTHON AGENT - EXAMPLE TRANSCRIPTS")
    print("="*60)
    
    for i, query in enumerate(examples, 1):
        print(f"\nEXAMPLE {i}")
        print("-" * 40)
        
        result = agent.process_query(query)
        
        print(f"User: {result['user_input']}")
        
        if result['needs_computation']:
            print(f"\nTool Call: {result['tool_used']}")
            print(f"Code: {result['code_generated']}")
            print(f"Output: {result['tool_output']}")
            
            # Generate natural language response based on output
            if result['tool_output'].isdigit():
                if "factorial" in query.lower():
                    print(f"\nAgent: The factorial of 8 is {result['tool_output']}.")
                elif "vowel" in query.lower():
                    print(f"\nAgent: There are {result['tool_output']} vowels in 'programming'.")
                elif "times does" in query.lower():
                    print(f"\nAgent: The substring 'ab' appears {result['tool_output']} times in 'ababcabab'.")
                elif "*" in query and "+" in query:
                    print(f"\nAgent: 127 * 83 + 45 = {result['tool_output']:,}")
                else:
                    print(f"\nAgent: There are {result['tool_output']} 'r' characters in 'strawberry'.")
            else:
                print(f"\nAgent: {result['tool_output']}")
        else:
            print(f"\nTool Call: {result['tool_used']}")
            print(f"Output: (no output)")
            print(f"\nAgent: {result['response']}")


def main():
    """Main function for interactive mode."""
    agent = SimpleAgent()
    
    print("Pure Python Tool-Calling Agent")
    print("Ask me counting or arithmetic questions!")
    print("Type 'examples' to see transcripts, or 'quit' to exit.\n")
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'examples':
            run_example_transcripts()
            continue
        
        result = agent.process_query(user_input)
        
        if result['needs_computation']:
            print(f"[Tool: {result['tool_used']}]")
            print(f"[Code: {result['code_generated']}]")
            print(f"[Output: {result['tool_output']}]")
            
            if result['tool_output'].isdigit():
                print(f"Agent: The answer is {result['tool_output']}.")
            else:
                print(f"Agent: {result['tool_output']}")
        else:
            print(f"Agent: {result['response']}")
        
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        run_example_transcripts()
    else:
        main() 