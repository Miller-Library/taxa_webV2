import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from openai.embedding import OpenAIEmbeddings
from faiss.vector import TokenTextSplitter, FAISS

# Function to remove citation pages from PDF
def remove_citation_pages(pdf_path):
    #Read the input file
    pdf_reader = PdfReader(pdf_path)

    #Initialize a new PDF writer
    new_pdf = PdfWriter()

    #Define the variations of citations to look for
    citation_variations = ["Works Cited", "References", "Citations", "Literature Cited", "Literature Citations"]
    #Flag to track if a citation is found
    citation_found = False

    #Iterate through each page in the input file
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() #Extract text from the page

        #Check if a citation variation is found
        if not citation_found:
            for variation in citation_variations:
                if variation in page_text or variation.upper() in page_text:
                    citation_found = True
                    break
        #If no citation is found, add the page to the new PDF
        if not citation_found:
            new_pdf.add_page(page)

    #Write the new PDF to a file
    new_pdf_path = "output.pdf"
    with open(new_pdf_path, "wb") as f:
        new_pdf.write(f)

    return new_pdf_path

# Function to read the content of a PDF
def read_new_pdf(new_pdf_path):
    pdf_text = ""

    with open(new_pdf_path, "rb") as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

    return pdf_text

# Function to get text from PDFs, split into chunks, and create vector store
def process_pdf_files(pdf_docs):
    pdf_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(pdf_text)
    vectorstore = get_vectorstore(text_chunks)
    return pdf_text, text_chunks, vectorstore

# Streamlit application
def main():
    st.title("PDF Citation Removal and Processing")

    # Upload a PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Display uploaded file details
        st.text(f"Uploaded file: {uploaded_file.name}")

        # Remove citation pages and generate new PDF
        output_pdf_path = remove_citation_pages(uploaded_file)

        # Read and display the content of the new PDF
        new_pdf_content = read_new_pdf(output_pdf_path)
        st.subheader("Content of the new PDF without citations:")
        st.write(new_pdf_content)

        # Process the new PDF for text chunks and vector store
        pdf_text, text_chunks, vectorstore = process_pdf_files([output_pdf_path])
        st.subheader("Processed PDF Text:")
        st.write(pdf_text)
        st.subheader("Text Chunks:")
        st.write(text_chunks)
        st.subheader("Vector Store:")
        st.write(vectorstore)

if __name__ == "__main__":
    main()
