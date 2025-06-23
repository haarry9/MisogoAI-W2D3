# Q2 Reward Model Summary

We trained a reward model using GPT-2 to capture preferences for 5 creative prompts. For each prompt, we generated 4 candidate answers and ranked them manually. Using Hugging Face's TRL RewardTrainer, we trained for 100 steps.

The model was later evaluated on a new prompt, and the reward scores showed a reasonable correlation between quality and predicted scores.

Challenges:
- Hard to rank similar answers.
- GPT-2 sometimes produces low-quality generations.

Next steps could include trying larger models or increasing dataset size for better alignment.
