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
        pages = [page.get_text() for page in doc]
        doc.close()

        total_pages = len(pages)
        text = "".join(pages)
        char_count = len(text)
        word_count = len(text.split())
        est_tokens = char_count // 4
        st.caption(f"📊 Extracted: {total_pages} pages, {char_count:,} characters (~{word_count:,} words, ~{est_tokens:,} tokens)")

        keywords = st.text_input(
            "🔑 Key topics/concepts (optional)",
            placeholder="e.g., photosynthesis, derivatives, dynamic programming",
            help="Enter important topics to focus flashcard generation"
        )

        CHUNK_SIZE = 3

        if st.button("Generate Flashcards"):
            model_id = os.getenv("FINE_TUNED_MODEL_ID", "gpt-4.1-mini-2025-04-14")

            try:
                system_msg = "You are a flashcard generator. Generate concise Q&A flashcards from the text provided by the user. Focus on the most important concepts and information."
                if keywords:
                    system_msg += f" Prioritize flashcards on these topics: {keywords}. Only generate a flashcard for a topic if there is direct supporting text; skip any topic not covered in the text."

                chunks = [pages[i:i + CHUNK_SIZE] for i in range(0, total_pages, CHUNK_SIZE)]
                all_flashcards = []
                progress = st.progress(0)

                for idx, chunk in enumerate(chunks):
                    chunk_text = "".join(chunk)
                    start_page = idx * CHUNK_SIZE + 1
                    end_page = min(start_page + CHUNK_SIZE - 1, total_pages)
                    progress.progress((idx + 1) / len(chunks), text=f"Processing pages {start_page}–{end_page} (chunk {idx + 1}/{len(chunks)})...")

                    response = client.responses.parse(
                        model=model_id,
                        input=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": chunk_text}
                        ],
                        text_format=FlashcardSet
                    )
                    all_flashcards.extend(response.output_parsed.flashcards)

                progress.empty()
                st.success(f"✅ Generated {len(all_flashcards)} flashcards across {len(chunks)} chunks!")

                anki_lines = []
                for card in all_flashcards:
                    front = card.question.replace(";", " -")
                    back = card.answer.replace(";", " -")
                    tags = " ".join(tag.replace(" ", "_") for tag in card.tags)
                    anki_lines.append(f"{front};{back};{tags}")

                st.download_button(
                    label="📥 Download for Anki",
                    data="\n".join(anki_lines),
                    file_name=f"{pdf.name[:-4]}_flashcards.txt",
                    mime="text/plain"
                )

                for card in all_flashcards:
                    with st.expander(f"❓ {card.question}"):
                        if card.tags:
                            st.markdown(" ".join([f"`{tag}`" for tag in card.tags]))
                        st.write(card.answer)

            except Exception as e:
                st.error(f"Error generating flashcards: {str(e)}")

if __name__ == "__main__":
    main()