# Training Data

`training_data.jsonl` contains supervised fine-tuning (SFT) data in OpenAI chat format. Each line is a JSON object with a `messages` array containing `system`, `user`, and `assistant` turns.

## Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a flashcard generator. Generate concise Q&A flashcards from the text provided by the user. Focus on the most important concepts and information."
    },
    {
      "role": "user",
      "content": "<source text passage>"
    },
    {
      "role": "assistant",
      "content": "{\"flashcards\":[{\"tags\":[\"accounting\",\"business\"],\"question\":\"What is accounting?\",\"answer\":\"Accounting is the process of organizing, analyzing, and communicating financial information used for decision-making.\"}]}"
    }
  ]
}
```

The assistant response is a JSON string with a `flashcards` array. Each flashcard has:
- `tags` — list of topic strings for Anki tagging
- `question` — the front of the flashcard
- `answer` — the back of the flashcard

## Distractor Keywords

Some examples include distractor keywords in the system prompt to train the model to generate flashcards only for topics with direct supporting text:

```
Prioritize flashcards on these topics: gestational diabetes screening, quad marker screen.
Only generate a flashcard for a topic if there is direct supporting text; skip any topic not covered in the text.
```

This teaches the model to skip topics not present in the passage, reducing hallucinated flashcards at inference time.