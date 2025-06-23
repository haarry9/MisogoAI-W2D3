"""
Test script for the Tool-Calling Agent
Tests the agent implementation with mock Python execution
"""

from agent import PythonExecTool, NoOpTool


def test_python_tool():
    """Test the Python execution tool with all example queries."""
    tool = PythonExecTool()
    
    print("TESTING PYTHON EXECUTION TOOL")
    print("="*50)
    
    # Test cases from the examples
    test_cases = [
        {
            "name": "Character counting - 'r' in 'strawberry'",
            "code": "result = len([c for c in 'strawberry' if c == 'r'])\nprint(result)",
            "expected": "3"
        },
        {
            "name": "Vowel counting in 'programming'",
            "code": "vowels = 'aeiouAEIOU'\nword = 'programming'\ncount = sum(1 for char in word if char in vowels)\nprint(count)",
            "expected": "3"
        },
        {
            "name": "Substring counting - 'ab' in 'ababcabab'",
            "code": "text = 'ababcabab'\nsubstring = 'ab'\ncount = 0\nfor i in range(len(text) - len(substring) + 1):\n    if text[i:i+len(substring)] == substring:\n        count += 1\nprint(count)",
            "expected": "4"
        },
        {
            "name": "Complex arithmetic - 127 * 83 + 45",
            "code": "result = 127 * 83 + 45\nprint(result)",
            "expected": "10586"
        },
        {
            "name": "Factorial of 8",
            "code": "import math\nresult = math.factorial(8)\nprint(result)",
            "expected": "40320"
        }
    ]
    
    passed = 0
    for i, test in enumerate(test_cases, 1):
        result = tool._run(test["code"])
        status = "✓ PASS" if result == test["expected"] else "✗ FAIL"
        print(f"Test {i}: {test['name']}")
        print(f"  Expected: {test['expected']}")
        print(f"  Got: {result}")
        print(f"  Status: {status}\n")
        
        if result == test["expected"]:
            passed += 1
    
    print(f"PYTHON TOOL TESTS: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_noop_tool():
    """Test the noop tool."""
    tool = NoOpTool()
    
    print("\nTESTING NOOP TOOL")
    print("="*50)
    
    result = tool._run("any input")
    expected = ""
    
    status = "✓ PASS" if result == expected else "✗ FAIL"
    print(f"Noop tool test:")
    print(f"  Expected: '{expected}'")
    print(f"  Got: '{result}'")
    print(f"  Status: {status}")
    
    return result == expected


def main():
    """Run all tests."""
    print("TOOL-CALLING AGENT TEST SUITE")
    print("Testing core tools functionality\n")
    
    python_passed = test_python_tool()
    noop_passed = test_noop_tool()
    
    print(f"\n{'='*60}")
    print("OVERALL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Python Tool: {'PASS' if python_passed else 'FAIL'}")
    print(f"Noop Tool: {'PASS' if noop_passed else 'FAIL'}")
    
    if python_passed and noop_passed:
        print("\n✓ All tests passed! The agent tools are working correctly.")
        print("Set your OPENAI_API_KEY environment variable and run 'python agent.py' to test the full agent.")
    else:
        print("\n✗ Some tests failed. Check the implementation.")
    
    print("\nTo see example transcripts, run: python examples.py")


if __name__ == "__main__":
    main() 