import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
import io
import openai
import base64
import fitz
from PIL import Image
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader, PdfWriter
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai  import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# #Set you OpenAI API key
# OPENAI_API_KEY = "your openai api key"
OPENAI_API_KEY  = os.environ['OPENAI_API_KEY']

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="750" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def initialize_session_state():
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = None

    if "pdf_image" not in st.session_state:
        st.session_state.pdf_image = None

    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None

def set_openai_api_key(api_key):
    st.session_state["openai_api_key"] = api_key

def sidebar():
    with st.sidebar:
        st.title('🦜️🔗Q&A with your PDF - Langchain & OpenAI Powered Chatbot')
        st.markdown('''
        How to use:
        1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑
        2. Upload a PDF File 📄 (Max Size: 200MB)
        3. Enter a question in the prompt bar.
        ''')
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=st.session_state.get("OPENAI_API_KEY", ""),
        )

        if api_key_input:
            set_openai_api_key(api_key_input)

        st.markdown("----")
        st.markdown('''
        # About
        This app utilizes the power of LangChain and OpenAI's powerful language model to provide a conversational Q&A chatbot.
        Users can upload a PDF document, and the chatbot will answer questions about the document's content.

        This tool is a work in progress.

        # Resources:

        - [Streamlit](https://streamlit.io/)
        - [Langchain](https://python.langchain.com/docs/use_cases/question_answering/)
        - [OpenAI](https://openai.com/)
        ''')

sidebar()
initialize_session_state()
#Formats the chat history string
def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)

#Identify the page of Literature Citations
def has_citations(page_text):
    # List of variations of "citations" to check for
    citation_variations = ["Works Cited", "References","Citations", "Literature Cited", "Literature Citations"]

    for variation in citation_variations:
        if variation in page_text:
            return True
    return False

def remove_citation_pages(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    new_pdf = PdfWriter()

    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()

        if not has_citations(page_text):
            new_pdf.add_page(page)

    new_pdf_path = "output.pdf"
    with open(new_pdf_path, "wb") as f:
        new_pdf.write(f)

    return new_pdf_path

# Snippet needs to be refined to capture multiple pages of citations. This block above
# only captures the first page with citation variations in the name
# def remove_citation_pages(pdf_path):
#     doc = fitz.open(pdf_path)
#     non_citation_pages = []
#
#     for page_num in range(doc.page_count):
#         page = doc[page_num]
#         page_text = page.get_text()
#
#         if not has_citations(page_text):
#             non_citation_pages.append(page_num)

    # # Create a new PDF with only non-citation pages
    # new_pdf = fitz.open()
    # for page_num in non_citation_pages:
    #     new_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
    #
    # return new_pdf


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

#Stream Lit UI
st.header("📄 PDF Q&A")
st.subheader("Load your PDF, ask questions, and receive answers from yourAd document.")

uploaded_file = st.file_uploader("Choose a PDF file", type = "pdf")

if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.write(file_details)


    st.text("Processing PDF File...⏳")
    #Read the PDF file
    doc_reader = PdfReader(uploaded_file)

    with open("input.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    new_pdf =  remove_citation_pages("input.pdf")

    st.success("PDF uploaded successfully with citation pages removed!")
    show_pdf(new_pdf)

    # Display the new PDF without citation pages
    extract_text = extract_text_from_pdf(new_pdf)
    # st.text_area("Extracted Text", extract_text)

    #load a tiktoken splitter directly

    tk_text_splitter = TokenTextSplitter(chunk_size = 800, chunk_overlap = 40)

    #Split text into chunk_size with chunk_overlap
    docs2 = tk_text_splitter.split_text(extract_text)

    #Initialize Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    #Load documents into vector database, FAISS
    #similarity search in the retriever object where
    #it selects text chunk vectors that are most similar to the question vector.
    # k =3 meaning we want the three most relevant chunks
    vector_store2 = FAISS.from_texts(docs2, embeddings).as_retriever(search_type = "similarity", search_kwargs= {"k": 3})


    #Initialize our large language model (llm)
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

    #iterates over the initially returned documents
    #and extract from each only the content that is relevant to the query.
    compressor = LLMChainExtractor.from_llm(llm)

    #Wrap our base retriever with ContextualCompressionRetriever
    compression_retriever = ContextualCompressionRetriever(base_compressor = compressor,base_retriever=vector_store2)


    # Build prompt
    custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Please use a maximum of 4 sentences. If you do not know the answer reply with 'I am sorry'.
    1. SPECIES NAME: The scientific name (species name) of any plant, animal, fungus, alga or bacterium consists of two Latinized words. The first word is the name of the genus to which the organism belongs. The second word is the specific epithet or specific term of the species. Together, the genus plus the specific epithet make up the species name. The species name and scientific name are synonyms.
    2. HABITAT: A species habitat can be seen as the physical manifestation of its ecological niche.
    3. LOCATION: Name of any geographic location, like cities, countries, continents, districts etc.

    Examples:

    1. Sentence: Strongylocentrotus fransiscanus and S. purpuratus were obtained from the subtidal and intertidal regions, respectively, of Monterey Bay.
    "Output: {{'SPECIES NAME': ['Strongylocentrotus fransiscanus', 'S. purpuratus'], 'HABITAT': ['subtidal', 'intertidal'], 'LOCATION': ['Monterey Bay']}}

    2. Sentence: Cucumaria curata and C. pseudocurata live and feed in the hydrodynamically stressful environment of exposed intertidal areas.
    Output: {{'SPECIES NAME': ['Cucumaria curata', 'C. pseudocurata'], 'HABITAT': ['exposed intertidal'], 'LOCATION': ['None']}}

    Chat History:
    {chat_history}
    Follow Up Input: {question}
"""
    #

    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    #This memory allows for storing of messages and then extracts the messages in a variable.
    memory = ConversationBufferMemory(
        memory_key = "chat_history", input_key = "question", output_key = "answer", return_messages = True)

    #Building the ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(),
                                                     compression_retriever,
                                                     memory = memory,
                                                     condense_question_prompt = CUSTOM_QUESTION_PROMPT,
                                                     get_chat_history = get_chat_history,
                                                     return_source_documents = True,
                                                     verbose = True
                                                     )

    #Initialize Chat History

    chat_history = []

#Get the user's query

    query = st.text_input(f'Enter prompt here about {uploaded_file.name}')


    #Streamlit For Generating Responses and display in app

    generate_button = st.button("Generate Response")

    if generate_button and query:
        with st.spinner("Generating response..."):
            result = qa_chain({"question": query, "chat_history": chat_history})

            answer = result["answer"]
            source_documents = result["source_documents"]
            chat_history = result["chat_history"]

            #Combine the answer and source documents into a single Response
            response = {
                "answer": answer,
                "source_documents": source_documents,
                "chat_history": chat_history
            }
            st.write("response:", response)
        with st.expander("Document Similarity Search"):
            search = vector_store2.vectorstore.similarity_search_with_score(query, k = 3)

            st.write(search[0][0].page_content)
