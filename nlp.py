import fitz  # PyMuPDF
from transformers import pipeline
import streamlit as st


#uoload and extract the pdf
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#preprocess and summarise the text
def summarize_text(text, max_chunk=1000):
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summary = ""
    for chunk in chunks:
        summary_chunk = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        summary += summary_chunk + " "
    return summary

#setting the title for the streamlit page
st.title("PDF Summarizer")

#upload the pdf file
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if pdf_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    raw_text = extract_text_from_pdf("temp.pdf")
    st.text_area("Extracted Text", raw_text[:2000], height=300)

    if st.button("Summarize"):
        summary = summarize_text(raw_text)
        st.subheader("Summary")
        st.write(summary)
