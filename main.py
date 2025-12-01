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

embeddings = OllamaEmbeddings(model = "snowflake-arctic-embed")

vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory="./chromadb",
    collection_name="simple-rag"
)

llm = ChatOllama(model='llama3.1', stream = False)



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

# Initialize session memory
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input box
user_input = st.chat_input("How can I help you with your insurance needs!")

if user_input:
    # Store user's message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Prepare history for LLM
    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state["messages"][:-1]]
    )

    # Generate and store assistant response (but DO NOT render it here)
    with st.spinner("Thinking…"):
        response = chain.invoke({
            "question": user_input,
            "history": history_text,
        })

    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.rerun()











# st.set_page_config(page_title="PolicyPilot")
# st.title("PolicyPilot")
# st.write("Your Personal Insurance Advisor")


# # Initialize session memory
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []


# # Display chat history
# for msg in st.session_state["messages"]:
#     if msg["role"] == "user":
#         with st.chat_message("user"):
#             st.write(msg["content"])
#     else:
#         with st.chat_message("assistant"):
#             st.write(msg["content"])



# user_input = st.chat_input("How can I help you with your insurance needs!")

# if user_input:
#     # Add user message to history
#     st.session_state["messages"].append({"role": "user", "content": user_input})

#     # Prepare conversation history as a text block
#     history_text = "\n".join(
#         [f"{m['role']}: {m['content']}" for m in st.session_state["messages"][:-1]]
#     )

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking…"):
#             response = chain.invoke({
#                 "question": user_input,
#                 "history": history_text,
#             })
#             st.write(response)

#     # Store assistant message
#     st.session_state["messages"].append({"role": "assistant", "content": response})
