import pymupdf
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import os

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Flashcard(BaseModel):
    tags: list[str]
    question: str
    answer: str

class FlashcardSet(BaseModel):
    flashcards: list[Flashcard]

with st.sidebar:
    st.title('FlashFormer')
    st.markdown('''
    ## About FlashFormer:

    Generate personalized flashcard-style study guides from your PDFs using OpenAI

    ''')
    st.space("medium")

def main():
    st.header("📚 Generate Flashcards from PDF")

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        st.write(f"Processing: {pdf.name}")

        doc = pymupdf.open(stream=pdf.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        char_count = len(text)
        word_count = len(text.split())
        est_tokens = char_count // 4
        st.caption(f"📊 Extracted: {char_count:,} characters (~{word_count:,} words, ~{est_tokens:,} tokens)")

        keywords = st.text_input(
            "🔑 Key topics/concepts (optional)",
            placeholder="e.g., photosynthesis, derivatives, dynamic programming",
            help="Enter important topics to focus flashcard generation"
        )

        if st.button("Generate Flashcards"):
            with st.spinner("Generating flashcards..."):
                model_id = os.getenv("FINE_TUNED_MODEL_ID", "gpt-4.1-mini-2025-04-14")

                try:
                    system_msg = "You are a flashcard generator. Generate concise Q&A flashcards from the text provided by the user. Focus on the most important concepts and information."
                    if keywords:
                        system_msg += f" Prioritize flashcards on these topics: {keywords}."

                    user_msg = text

                    response = client.responses.parse(
                        model=model_id,
                        input=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg}
                        ],
                        text_format=FlashcardSet
                    )

                    flashcard_set = response.output_parsed
                    st.success(f"✅ Generated {len(flashcard_set.flashcards)} flashcards!")

                    for card in flashcard_set.flashcards:
                        with st.expander(f"❓ {card.question}"):
                            if card.tags:
                                st.markdown(" ".join([f"`{tag}`" for tag in card.tags]))
                            st.write(card.answer)

                except Exception as e:
                    st.error(f"Error generating flashcards: {str(e)}")

if __name__ == "__main__":
    main()