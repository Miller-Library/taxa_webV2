import streamlit as st
from streamlit_chat import message
import base64
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from html_template import css, bot_template, user_template
from langchain import PromptTemplate



def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="750" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def remove_citation_pages(pdf_path):
    #Read the input file
    pdf_reader = PdfReader(pdf_path)

    #Initialize a new PDF writer
    new_pdf = PdfWriter()

    #Define the variations of citations to look for
    citation_variations = ["Works Cited", "References", "Citations", "Literature Cited", "Literature Citations"]
    #Flag to track if a citation is found
    citation_found = False

    processed_text = []
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
            processed_text.append(page_text)

    return '/n'.join(processed_text)
    #Write the new PDF to a file
    # new_pdf_path = "output.pdf"
    # with open(new_pdf_path, "wb") as f:
    #     new_pdf.write(f)
    #
    # return new_pdf_path


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Read in each PDF and extract text from its pages
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def fix_line_breaks(text):
    # Join lines that have been incorrectly split
    fixed_text = text.replace('\n', ' ')

    # More line break fixing tasks can be added here

    return fixed_text

    #Split the text per chunk_size and chunk_overlap using TokenTextSplitter
def get_text_chunks(text):
    text_splitter = TokenTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

    #Create embeddings
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Please use a maximum of 4 sentences. If you do not know the answer reply with 'I am sorry'.
    1. SPECIES NAME: The scientific name (species name) of any plant, animal, fungus, alga or bacterium consists of two Latinized words. The first word is the name of the genus to which the organism belongs. The second word is the specific epithet or specific term of the species. Together, the genus plus the specific epithet make up the species name. The species name and scientific name are synonyms.
    2. HABITAT: A species habitat can be seen as the physical manifestation of its ecological niche.
    3. LOCATION: Name of any geographic location, like cities, countries, continents, districts etc.
    4. COORDINATE: geographic coordinates are numerical values that pinpoint a specific location or position within a space, such as a map, graph, or three-dimensional environment. They provide a way to precisely describe where something is situated by using a set of values, usually represented as ordered pairs or triples.
    5. DEPTH: Some species live in water environments so a depth may be provided which is usually a numerical value expressed in either metric: meter or foot

    Example of COORDINATES separated by "|": 48°51'30.24″N, 2°17'40.2″E | 41°24'12.2"N 2°10'26.5"E

    Chat History:
    {chat_history}
    Follow Up Input: {question}
"""
    #
    chat_history = []
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', input_key = "question", output_key = "answer", return_messages=True)
    # conversation_chain = load_qa_chain(llm, chain_type = "stuff", memory = memory, prompt = CUSTOM_QUESTION_PROMPT)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        condense_question_prompt = CUSTOM_QUESTION_PROMPT,
        return_source_documents = True,
        verbose = True
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.session_state.document_chunks = response['source_documents']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0: #Every even number will be user input
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:# every odd number will be AI response
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


    with st.expander("Source Documents"): #Shows most relevant documents for generated response
        st.write(response["source_documents"])


def main():
    load_dotenv()
    st.set_page_config(page_title="Q&A with your PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("<h1 style = 'text-align: center;'>Chat with multiple PDFs 📄: Powered by Langchain and OpenAI 🦜️🔗</h1>", unsafe_allow_html=True)
    user_question = st.text_input("Ask questions about your documents:")

    if user_question:
        handle_userinput(user_question)

    uploaded_files = st.file_uploader(
    "Upload your PDFs here. Click 'Process' to ingest files.", accept_multiple_files=True)

    if uploaded_files:
        st.button("Remove Bibliography")

        for uploaded_file in uploaded_files:
            # read_df = PdfReader(uploaded_file)
            new_pdf_content = remove_citation_pages(uploaded_file)
            # processed_text = new_pdf_content.decode("utf-8")
            new_text = fix_line_breaks(new_pdf_content)
            st.write(new_text)
    with st.sidebar:

            #Spinner for UI to verify it's running
        with st.spinner("Processing files..."):
            #if the Process button is pushed
            if st.button("Process"):

                #extracts the remaining text out from PDF
                raw_text = get_pdf_text(new_text)

                # raw_text2 = fix_line_breaks(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                #Display successful message once done
                st.success("Embeddings Done", icon="✅")

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)



if __name__ == '__main__':
    main()
