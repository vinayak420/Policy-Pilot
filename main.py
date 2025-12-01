from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OllamaEmbeddings(model = "snowflake-arctic-embed")

vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory="./chromadb",
    collection_name="simple-rag"
)

llm = ChatOllama(model='llama3.1')



QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
    )

retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

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
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


#--------------UI------------

st.title("PolicyPilot — Your Personal Insurance Advisor")
st.write("Ask anything about insurance documents you've uploaded.")

user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching database and generating response..."):
            answer = chain.invoke({"question": user_question})
        st.success("Answer:")
        st.write(answer)
