import base64
import os
import time
from tempfile import NamedTemporaryFile
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain import PromptTemplate
import os

# Load environment variables
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()

if not os.path.exists("./data"):
    os.makedirs("./data")

st.set_page_config(
    page_title="RAG with Gemini",
    page_icon=":gemini:",
    layout="centered",
)

def display_pdf(document):
    with open(document, "rb") as file:
        pdf_64 = base64.b64encode(file.read()).decode("utf-8")
        display = f'<iframe src="data:application/pdf;base64,{pdf_64}"style="overflow: hidden; width: 100%; height: 450px; border-radius: 35px;"></iframe>'
        st.markdown(display, unsafe_allow_html=True)

# hide = """
#     <style>
#     #MainMenu{visibility: hidden;}
#     footer{visibilty: hidden;}
#     header{visibility: hidden;}
#     </style>

#     """
# st.markdown(hide, unsafe_allow_html=True)

header_html = """
    <style>
        /* Style for the header */
        .header {
    
            color: white;
            text-decoration: none; /* Remove underlining */
            text-shadow: -1px -1px 0 black, 1px -1px 0 black, -1px 1px 0 black, 1px 1px 0 black; /* Black outline */
        }

        /* Style for hover effect */
        .header:hover {
            letter-spacing: 2px;
            color: #C683D7;
            cursor: pointer;
            transition: 0.2s;
        
        }
    </style>

    <!-- Header HTML -->
    <h1 class="header">♊ RAGemini</h1>
"""


# Display HTML
st.markdown(header_html, unsafe_allow_html=True)

template = """Use the following pieces of context to answer the task or question at the end. 
If you don't know the response, don't try to make up anything. 
If there is no question or task at the end and just a statement like 'good' , 'bad' etc. , then provide a response to the statement appropriately.
If there is seemingly no context for the question, respond with a general according to the context provided.
At the end of the document give a brief source of the answer within the document.
Keep the answer as concise and detailed as possible.
Try to keep  the answers relatively longer.
Keep a cheery tone overall and try to keep the answers as positive as possible.
{context}
Question: {question}
Helpful Answer:"""


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)

gen_ai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True,
)
doc_loader = st.file_uploader("Upload a document", type=["pdf"])
if doc_loader:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(doc_loader.read())
        temp_file_path = temp_file.name

    # Use the temporary file path with PyPDFLoader
    doc = PyPDFLoader(temp_file_path)
    pages = doc.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(
        search_kwargs={"k": 4}
    )
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    st.success("RAG Setup Successfully")
    if st.checkbox("Display Document"):
        display_pdf(temp_file_path)
        
    
    question = st.text_input("Ask Question")
    if question:
        response_placeholder = st.empty()
        with st.spinner("Searching the document..."):
            response = qa_chain({"query": question})
            response_text = response["result"]

        # Simulate streaming effect
        for i in range(len(response_text)):
            time.sleep(0.005)
            response_placeholder.write(response_text[: i + 1])

else:
    st.info("Upload a pdf to start RAG")

#         full_response = ""
#         assistant_response = response

#         for chunk in response:

#             for ch in chunk.text.split(" "):
#                 full_response += ch + " "
#                 time.sleep(0.05)

#                 message_placeholder.write(full_response + "|")

#         message_placeholder.write(full_response)
