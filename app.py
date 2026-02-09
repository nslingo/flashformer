import pymupdf
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

        # Extract text from PDF
        doc = pymupdf.open(stream=pdf.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # Generate flashcards button
        if st.button("Generate Flashcards"):
            with st.spinner("Generating flashcards..."):
                model_id = os.getenv("FINE_TUNED_MODEL_ID", "gpt-3.5-turbo")

                try:
                    response = client.responses.create(
                        model=model_id,
                        instructions="You are a flashcard generator that produces concise Q&A flashcards from the provided text. Format each as 'Q: [question]\nA: [answer]\n\n'",
                        input=f"Create flashcards from this text:\n\n{text[:4000]}"  # Limit tokens
                    )

                    flashcards_text = response.output_text

                    # Parse and display flashcards
                    flashcards = flashcards_text.strip().split("\n\n")
                    st.success(f"Generated {len(flashcards)} flashcards!")

                    for card in flashcards:
                        if "Q:" in card and "A:" in card:
                            parts = card.split("\nA:")
                            question = parts[0].replace("Q:", "").strip()
                            answer = parts[1].strip() if len(parts) > 1 else ""

                            with st.expander(f"❓ {question}"):
                                st.write(answer)

                except Exception as e:
                    st.error(f"Error generating flashcards: {str(e)}")


if __name__ == "__main__":
    main()