# FlashFormer

Turn your PDFs into Anki-ready flashcards using a fine-tuned OpenAI model.

## Features

- **SFT fine-tuning** - uses a supervised fine-tuned model trained on flashcard Q&A pairs to improve flashcard quality and consistency
- **3-page chunking** - processes PDFs in chunks to handle large documents while reducing hallucination and avoiding hitting token limits
- **Distractor keywords in training data** - teaches the model to generate flashcards only for topics supported by the text, reducing hallucinated cards
- **Anki export** - downloads flashcards as a semicolon-delimited `.txt` file ready to import into Anki

## Installation

```bash
git clone https://github.com/nslingo/flashformer.git
cd flashformer
pip install -r requirements.txt
```
Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_key_here
FINE_TUNED_MODEL_ID=your_fine_tuned_model_id  # optional, defaults to gpt-4.1-mini-2025-04-14
```

To use the fine-tuned model, upload `training/training_data.jsonl` and fine-tune following the [OpenAI SFT guide](https://developers.openai.com/api/docs/guides/supervised-fine-tuning/). Once complete, set the resulting model ID in your `.env` as `FINE_TUNED_MODEL_ID`. 

## Usage

```bash
streamlit run app.py
```

Upload a PDF, optionally enter key topics, and click **Generate Flashcards**. Download the result to import into Anki via **File > Import**.