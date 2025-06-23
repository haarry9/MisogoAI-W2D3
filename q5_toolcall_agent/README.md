# Tool-Calling Agent: Overcoming Tokenizer Blindness

A minimal agent implementation that uses Python execution to provide accurate counting and arithmetic capabilities, solving the "tokenizer blindness" problem where LLMs struggle with precise character counting.

## 🎯 Problem Statement

Large Language Models (LLMs) suffer from **tokenizer blindness** - they cannot accurately count characters, substrings, or perform precise arithmetic because they process text as tokens rather than individual characters. For example:

- **Question**: "How many 'r' characters are in 'strawberry'?"
- **LLM Response**: Often incorrect (might say 2 instead of 3)
- **Root Cause**: The word "strawberry" gets tokenized, losing character-level information

## 💡 Solution

This project implements a **tool-calling agent** that:
1. **Detects** when queries require precise computation
2. **Generates** appropriate Python code
3. **Executes** the code in a safe environment
4. **Returns** accurate results

### Architecture

```
User Query → Agent → Tool Selection → Python Execution → Accurate Result
```

## 📁 Project Structure

```
q5_toolcall_agent/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── agent.py              # LangChain-based agent implementation
├── pure_python_agent.py  # Standalone Python implementation
├── examples.py           # Example transcripts (3 counting + 2 arithmetic)
├── test_agent.py         # Test suite for tool functionality
```

## 📋 File Descriptions

### `agent.py` - LangChain Implementation
**Main agent using LangChain framework**

- **`PythonExecTool`**: Executes Python code in temporary files, returns stdout
- **`NoOpTool`**: Returns nothing (for normal, non-computational responses)
- **`ToolCallingAgent`**: Coordinates tool selection and execution
- **Features**: Full LangChain integration, OpenAI GPT-4, robust error handling

### `pure_python_agent.py` - Standalone Implementation
**Self-contained agent without external dependencies**

- **`PythonExecutor`**: Core Python execution engine
- **`SimpleAgent`**: Pattern-based query analysis and code generation
- **Features**: No LangChain required, regex-based pattern matching, lightweight

### `examples.py` - Example Transcripts
**Demonstrates the 5 required examples with full transcripts**

**Counting Examples (3):**
1. Character counting: "How many 'r' in 'strawberry'?" → 3
2. Vowel counting: "Count vowels in 'programming'" → 3  
3. Substring counting: "How many times does 'ab' appear in 'ababcabab'?" → 4

**Arithmetic Examples (2):**
4. Complex arithmetic: "What is 127 * 83 + 45?" → 10,586
5. Factorial: "Calculate the factorial of 8" → 40,320

### `test_agent.py` - Test Suite
**Validates tool functionality without requiring API keys**

- Tests Python execution tool with all example queries
- Verifies NoOp tool behavior
- Provides pass/fail status for each test case

### `requirements.txt` - Dependencies
**Python packages needed for LangChain implementation**

```
langchain==0.1.20
langchain-openai==0.1.8
langchain-community==0.0.38
openai==1.30.1
```

## 🚀 How to Run

### Option 1: View Example Transcripts (No setup required)
```bash
python examples.py
```
**Output**: Formatted transcripts showing expected agent behavior

### Option 2: Test Tool Functionality (No API key required)
```bash
python test_agent.py
```
**Output**: Test results verifying Python execution works correctly

### Option 3: Run Pure Python Agent
```bash
# Interactive mode
python pure_python_agent.py

# View examples
python pure_python_agent.py examples
```

### Option 4: Run LangChain Agent (Requires OpenAI API Key)
```bash
# Set your API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Install dependencies
pip install -r requirements.txt

# Run interactive agent
python agent.py
```

## 🔧 Usage Examples

### Interactive Session
```
User: How many 'r' in 'strawberry'?

Tool Call: python_exec
Code: len([c for c in 'strawberry' if c == 'r'])
Output: 3

Agent: There are 3 'r' characters in 'strawberry'.
```

### Arithmetic Example
```
User: What is 127 * 83 + 45?

Tool Call: python_exec  
Code: result = 127 * 83 + 45
Output: 10586

Agent: 127 * 83 + 45 = 10,586
```

## 🛡️ Safety Features

- **Sandboxed Execution**: Python code runs in temporary files
- **Timeout Protection**: 10-second execution limit
- **Error Handling**: Graceful error messages for invalid code
- **File Cleanup**: Automatic cleanup of temporary files

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_agent.py
```

Expected output:
```
✓ Test 1: Character counting - 'r' in 'strawberry' → 3
✓ Test 2: Vowel counting in 'programming' → 3  
✓ Test 3: Substring counting - 'ab' in 'ababcabab' → 4
✓ Test 4: Complex arithmetic - 127 * 83 + 45 → 10586
✓ Test 5: Factorial of 8 → 40320

PYTHON TOOL TESTS: 5/5 passed
```

## 🎯 Key Benefits

1. **Accuracy**: 100% accurate counting and arithmetic (vs LLM guessing)
2. **Transparency**: Shows exact Python code executed  
3. **Flexibility**: Handles complex patterns and edge cases
4. **Safety**: Sandboxed execution with timeout protection
5. **Simplicity**: Minimal implementation, easy to understand

## 🚀 Next Steps

- Add more mathematical operations (trigonometry, statistics)
- Implement data visualization capabilities  
- Add file I/O operations for larger datasets
- Create web interface for easier interaction

## 📝 License

This project is part of an educational assignment demonstrating tool-calling agents and overcoming LLM limitations. 