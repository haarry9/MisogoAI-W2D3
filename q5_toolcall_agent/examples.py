"""
Example transcripts for the Tool-Calling Agent
Demonstrates 3 counting queries and 2 arithmetic queries
"""

def print_transcript(title: str, user_input: str, tool_call: str, tool_output: str, final_response: str):
    """Print a formatted transcript of an interaction."""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {title}")
    print(f"{'='*60}")
    print(f"User: {user_input}")
    print(f"\nTool Call: python_exec")
    print(f"Code: {tool_call}")
    print(f"Output: {tool_output}")
    print(f"\nAgent: {final_response}")
    print(f"{'='*60}\n")


def main():
    """Show all 5 example transcripts."""
    
    print("TOOL-CALLING AGENT EXAMPLES")
    print("Demonstrating precise counting and arithmetic using Python execution")
    
    # COUNTING EXAMPLES (3)
    
    # Example 1: Character counting in "strawberry"
    print_transcript(
        title="Character Counting - 'r' in 'strawberry'",
        user_input="How many 'r' in 'strawberry'?",
        tool_call="len([c for c in 'strawberry' if c == 'r'])\nprint(len([c for c in 'strawberry' if c == 'r']))",
        tool_output="3",
        final_response="There are 3 'r' characters in 'strawberry'."
    )
    
    # Example 2: Vowel counting
    print_transcript(
        title="Vowel Counting - vowels in 'programming'",
        user_input="Count the number of vowels in 'programming'",
        tool_call="vowels = 'aeiouAEIOU'\nword = 'programming'\ncount = sum(1 for char in word if char in vowels)\nprint(count)",
        tool_output="3",
        final_response="There are 3 vowels in 'programming' (o, a, i)."
    )
    
    # Example 3: Substring counting
    print_transcript(
        title="Substring Counting - 'ab' in 'ababcabab'",
        user_input="How many times does 'ab' appear in 'ababcabab'?",
        tool_call="text = 'ababcabab'\nsubstring = 'ab'\ncount = 0\nfor i in range(len(text) - len(substring) + 1):\n    if text[i:i+len(substring)] == substring:\n        count += 1\nprint(count)",
        tool_output="4",
        final_response="The substring 'ab' appears 4 times in 'ababcabab'."
    )
    
    # ARITHMETIC EXAMPLES (2)
    
    # Example 4: Complex arithmetic
    print_transcript(
        title="Complex Arithmetic - multiplication and addition",
        user_input="What is 127 * 83 + 45?",
        tool_call="result = 127 * 83 + 45\nprint(result)",
        tool_output="10586",
        final_response="127 * 83 + 45 = 10,586"
    )
    
    # Example 5: Factorial calculation
    print_transcript(
        title="Factorial Calculation - 8!",
        user_input="Calculate the factorial of 8",
        tool_call="import math\nresult = math.factorial(8)\nprint(result)",
        tool_output="40320",
        final_response="The factorial of 8 is 40,320."
    )
    
    print("\nAll examples demonstrate how the agent uses Python execution to overcome")
    print("tokenizer blindness and provide accurate counting and arithmetic results.")


if __name__ == "__main__":
    main() 