import json
import re
import matplotlib.pyplot as plt
from collections import Counter
import openai
import os
from typing import List, Dict, Union, Tuple
import numpy as np

class ThinkingModeSampler:
    def __init__(self, api_key: str = None):
        """Initialize the sampler with OpenAI API key."""
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    def load_problems(self, filename: str = "problems.json") -> List[Dict]:
        """Load GRE-style problems from JSON file."""
        with open(filename, 'r') as f:
            return json.load(f)
    
    def create_chain_of_thought_prompt(self, problem: str) -> str:
        """Create a prompt that encourages step-by-step reasoning."""
        prompt = f"""Solve this math problem step by step.

Problem: {problem}

Let's think step-by-step:"""
        return prompt
    
    def generate_completion(self, prompt: str, temperature: float = 1.1, max_tokens: int = 150) -> str:
        """Generate a single completion using OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating completion: {e}")
            return ""
    
    def extract_numerical_answer(self, completion: str) -> Union[float, str, None]:
        """Extract the final numerical answer from completion text."""
        # Look for common answer patterns
        patterns = [
            r"(?:the answer is|answer:|final answer:|therefore,?)\s*([+-]?\d+(?:\.\d+)?)",
            r"([+-]?\d+(?:\.\d+)?)\s*(?:hours?|minutes?|dollars?|feet|inches|miles|years?|students?|percent|%)",
            r"(?:=|is|equals?)\s*([+-]?\d+(?:\.\d+)?)",
            r"([+-]?\d+(?:\.\d+)?)\s*$",  # Number at end of text
            r"(?:ratio|is)\s*(\d+:\d+)",  # For ratio answers
        ]
        
        completion_lower = completion.lower()
        
        # Special handling for ratio answers
        ratio_match = re.search(r"(\d+:\d+)", completion)
        if ratio_match:
            return ratio_match.group(1)
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, completion_lower, re.IGNORECASE)
            if matches:
                try:
                    # Return the last match (often the final answer)
                    answer = matches[-1]
                    return float(answer)
                except ValueError:
                    continue
        
        # If no pattern matches, look for any number in the text
        numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", completion)
        if numbers:
            try:
                return float(numbers[-1])  # Return last number found
            except ValueError:
                pass
        
        return None
    
    def majority_vote(self, answers: List[Union[float, str, None]]) -> Union[float, str, None]:
        """Perform majority vote on extracted answers."""
        # Filter out None values
        valid_answers = [ans for ans in answers if ans is not None]
        
        if not valid_answers:
            return None
        
        # Count occurrences
        counter = Counter(valid_answers)
        
        # Return most common answer
        return counter.most_common(1)[0][0]
    
    def self_consistency_inference(self, problem: str, num_samples: int = 10) -> Tuple[Union[float, str, None], List[str]]:
        """
        Perform self-consistency inference on a single problem.
        Returns: (majority_vote_answer, list_of_completions)
        """
        prompt = self.create_chain_of_thought_prompt(problem)
        completions = []
        answers = []
        
        for i in range(num_samples):
            completion = self.generate_completion(prompt, temperature=1.1)
            completions.append(completion)
            
            answer = self.extract_numerical_answer(completion)
            answers.append(answer)
            
            print(f"Sample {i+1}: {answer}")
        
        majority_answer = self.majority_vote(answers)
        return majority_answer, completions
    
    def deterministic_inference(self, problem: str) -> Union[float, str, None]:
        """Perform single deterministic inference (temperature=0)."""
        prompt = self.create_chain_of_thought_prompt(problem)
        completion = self.generate_completion(prompt, temperature=0)
        return self.extract_numerical_answer(completion)
    
    def normalize_answer(self, answer: Union[float, str, None], correct_answer: Union[float, str]) -> bool:
        """Check if extracted answer matches correct answer."""
        if answer is None:
            return False
        
        # Handle string answers (like ratios)
        if isinstance(correct_answer, str):
            return str(answer) == correct_answer
        
        # Handle numerical answers with tolerance
        try:
            if isinstance(answer, str) and ":" in answer:
                return False  # String ratio doesn't match number
            
            answer_float = float(answer)
            correct_float = float(correct_answer)
            
            # Allow small numerical tolerance
            return abs(answer_float - correct_float) < 0.01
        except (ValueError, TypeError):
            return False
    
    def evaluate_methods(self, problems: List[Dict]) -> Dict[str, float]:
        """Evaluate both methods on all problems and return accuracy scores."""
        sc_correct = 0  # Self-consistency correct
        det_correct = 0  # Deterministic correct
        total = len(problems)
        
        results = {
            'problems': [],
            'self_consistency_answers': [],
            'deterministic_answers': [],
            'correct_answers': []
        }
        
        for i, problem_data in enumerate(problems):
            problem = problem_data['problem']
            correct_answer = problem_data['answer']
            
            print(f"\n--- Problem {i+1} ---")
            print(f"Problem: {problem}")
            print(f"Correct answer: {correct_answer}")
            
            # Self-consistency method
            print("\nSelf-consistency sampling:")
            sc_answer, completions = self.self_consistency_inference(problem)
            print(f"Majority vote answer: {sc_answer}")
            
            # Deterministic method
            print("\nDeterministic inference:")
            det_answer = self.deterministic_inference(problem)
            print(f"Deterministic answer: {det_answer}")
            
            # Check correctness
            sc_is_correct = self.normalize_answer(sc_answer, correct_answer)
            det_is_correct = self.normalize_answer(det_answer, correct_answer)
            
            if sc_is_correct:
                sc_correct += 1
            if det_is_correct:
                det_correct += 1
            
            print(f"Self-consistency correct: {sc_is_correct}")
            print(f"Deterministic correct: {det_is_correct}")
            
            # Store results
            results['problems'].append(problem)
            results['self_consistency_answers'].append(sc_answer)
            results['deterministic_answers'].append(det_answer)
            results['correct_answers'].append(correct_answer)
        
        sc_accuracy = sc_correct / total
        det_accuracy = det_correct / total
        
        print(f"\n--- Final Results ---")
        print(f"Self-consistency accuracy: {sc_accuracy:.2%} ({sc_correct}/{total})")
        print(f"Deterministic accuracy: {det_accuracy:.2%} ({det_correct}/{total})")
        
        return {
            'self_consistency_accuracy': sc_accuracy,
            'deterministic_accuracy': det_accuracy,
            'results': results
        }
    
    def plot_accuracy_comparison(self, accuracy_data: Dict[str, float], save_path: str = "accuracy.png"):
        """Create and save accuracy comparison plot."""
        methods = ['Deterministic\n(temp=0)', 'Self-Consistency\n(temp=1.1, n=10)']
        accuracies = [
            accuracy_data['deterministic_accuracy'],
            accuracy_data['self_consistency_accuracy']
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, accuracies, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy Comparison: Deterministic vs Self-Consistency\nGRE Arithmetic Word Problems', 
                 fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        # Add horizontal lines for reference
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Accuracy comparison plot saved to {save_path}")

def main():
    """Main function to run the evaluation."""
    # Initialize sampler
    # Note: You need to set your OpenAI API key as environment variable OPENAI_API_KEY
    # or pass it directly: sampler = ThinkingModeSampler(api_key="your-key-here")
    
    try:
        sampler = ThinkingModeSampler()
        
        # Load problems
        problems = sampler.load_problems()
        print(f"Loaded {len(problems)} problems")
        
        # Evaluate both methods
        accuracy_data = sampler.evaluate_methods(problems)
        
        # Create accuracy comparison plot
        sampler.plot_accuracy_comparison(accuracy_data)
        
        return accuracy_data
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your OpenAI API key as environment variable OPENAI_API_KEY")
        return None

if __name__ == "__main__":
    results = main()
