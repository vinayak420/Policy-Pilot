from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import tempfile
from langchain_community.vectorstores import FAISS

if "uploaded_retriever" not in st.session_state:
    st.session_state["uploaded_retriever"] = None

embeddings = OllamaEmbeddings(model = "snowflake-arctic-embed")

#--------------Load vector DB

vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory="./chromadb",
    collection_name="simple-rag"
)

llm = ChatOllama(model='llama3.1', stream = False)


#---------document upload------------

st.sidebar.title("Upload Policy Documents")

uploaded_files = st.sidebar.file_uploader(

    "Upload",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.info("Processing")

    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(pdf.read())
            temp_path = temp.name
        loader = PyPDFLoader(temp_path)
        docs.extend(loader.load())
    
    chunks = text_splitter.split_documents(docs)

    #FAISS store
    st.session_state["uploaded_retriever"] = FAISS.from_documents(
        chunks,
        embeddings
    ).as_retriever(search_kwargs={"k":10})

    st.success("Document uploaded!")

# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI language model assistant. Your task is to generate five
#     different versions of the given user question to retrieve relevant documents from
#     a vector database. By generating multiple perspectives on the user question, your
#     goal is to help the user overcome some of the limitations of the distance-based
#     similarity search. Provide these alternative questions separated by newlines.
#     Original question: {question}""",
#     )

# retriever = MultiQueryRetriever.from_llm(
#         vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
#     )






retriever = vector_db.as_retriever(search_kwargs = {"k":25})

template = """You are PolicyPilot, an insurance expert.

            Answer the question naturally and confidently. 
            Use the provided context ONLY IF it is relevant.
            If the context does not contain the answer, rely on your general knowledge.

            DO NOT mention the words “context”, “document”, “PDF”, or “according to the text”.
            DO NOT mention the provided documents.
            DO NOT say there is not enough information.
            Always provide a helpful and complete answer.
            DO not mention the question in your answer.
            {context}
            Question: {question}
            """

prompt = ChatPromptTemplate.from_template(template)

chain = (
        {
          "question": lambda x: x["question"],
          "context": lambda x: retriever.invoke(x["question"])
          }
        | prompt
        | llm
        | StrOutputParser()
    )


# --------------UI------------
st.set_page_config(page_title="PolicyPilot")
st.title("PolicyPilot")
st.write("Your Personal Insurance Advisor")

#session memory
if "messages" not in st.session_state:
    st.session_state["messages"] = []

#chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

#Input box
user_input = st.chat_input("How can I help you with your insurance needs!")

if user_input:
    # Store messages
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Prepare history for LLM
    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state["messages"][:-1]]
    )

    #Generate and store assistant response
    with st.spinner("Thinking…"):
        response = chain.invoke({
            "question": user_input,
            "history": history_text,
        })

    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.rerun()







